"""
train_client.py — Game.RL_Server 기반 병렬 RL 학습 클라이언트

M개 RL_Server × N개 에이전트 = M×N 동시 게임이 하나의 모델을 함께 학습.
단일 Python 프로세스 + 단일 asyncio 이벤트 루프.

========== 병렬화 구조 ==========

  M개 RL_Server 프로세스 (각각 다른 포트)
       ×
  N개 Python 비동기 에이전트 (asyncio 코루틴)
  ─────────────────────────────────────
  총 M × N 동시 게임 에피소드 → 하나의 모델 업데이트

Python 프로세스 수 = 1 (단일 학습 프로세스)
RL_Server 프로세스 수 = M
총 프로세스 수 = M + 1

========== 사용법 ==========

  단일 서버 모드 (RL_Server가 이미 포트 9000에서 실행 중):
    python train_client.py --port 9000 --n-agents 32

  다중 서버 모드 (M개 서버가 이미 실행 중):
    python train_client.py --multi --m-servers 4 --n-agents 32 --base-port 9000

  RL_Server 시작 방법 (M=4, 포트 9000~9003):
    for i in {0..3}; do dotnet run --project Game.RL_Server -- $((9000+i)) & done

========== 권장 설정 ==========

  M2 Pro (10코어):  M=4, N=24  → 96 동시 게임
  Ultra 7 265K (20코어): M=8, N=32  → 256 동시 게임
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import random
import shutil
import queue
import struct
import sys
import threading
import time
import traceback
from zipfile import ZIP_DEFLATED, ZipFile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
RUN_TAG: Optional[str] = None
ARCHIVE_RESULTS: bool = True


def _sanitize_tag(tag: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in tag.strip())
    return cleaned.strip('_') or 'run'


def _set_run_context(run_tag: Optional[str], archive_results: bool) -> str:
    global RUN_TAG, ARCHIVE_RESULTS
    ARCHIVE_RESULTS = bool(archive_results)
    if run_tag:
        RUN_TAG = _sanitize_tag(run_tag)
    else:
        RUN_TAG = _sanitize_tag(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    return RUN_TAG


def _apply_run_namespace(path: Path) -> Path:
    if RUN_TAG is None or path.is_absolute():
        return path
    parts = path.parts
    if not parts:
        return path
    if parts[0] in {'log', 'models'}:
        if len(parts) == 1:
            return Path(parts[0]) / RUN_TAG
        return Path(parts[0]) / RUN_TAG / Path(*parts[1:])
    return path


def _archive_dir(src: Path, dst_zip: Path) -> None:
    if not src.exists():
        return
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    if dst_zip.exists():
        dst_zip.unlink()
    with ZipFile(dst_zip, 'w', compression=ZIP_DEFLATED) as zf:
        if src.is_file():
            zf.write(src, arcname=src.name)
            return
        for file in src.rglob('*'):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(src.parent)))


def _archive_run_artifacts() -> None:
    if not ARCHIVE_RESULTS or RUN_TAG is None:
        return
    log_dir = Path(_resolve_project_path(f'log/{RUN_TAG}'))
    models_dir = Path(_resolve_project_path(f'models/{RUN_TAG}'))
    log_zip = Path(_resolve_project_path(f'log_{RUN_TAG}.zip'))
    models_zip = Path(_resolve_project_path(f'models_{RUN_TAG}.zip'))
    _archive_dir(log_dir, log_zip)
    _archive_dir(models_dir, models_zip)
    print(f'[*] archived logs:   {log_zip}')
    print(f'[*] archived models: {models_zip}')

# ---------------------------------------------------------------------------
# 로깅 (start.py 패턴)
# ---------------------------------------------------------------------------

class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _setup_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f'[*] log file: {log_file}')


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{whole_seconds:02d} ({seconds:.1f}s)'


def _raise_worker_errors(results: Sequence[Dict[str, Any]], context: str) -> None:
    errors = [r for r in results if r.get('__worker_error__')]
    if not errors:
        return
    details = []
    for err in errors:
        details.append(
            f"port={err.get('port')} error={err.get('error')}\n"
            f"{err.get('traceback', '')}"
        )
    raise RuntimeError(f'{context} worker failed:\n' + '\n'.join(details))


# ---------------------------------------------------------------------------
# Binary Protocol (RLProtocol.cs 와 동일해야 함)
# ---------------------------------------------------------------------------

HANDLER_INIT       = 1
HANDLER_STEP       = 2
HANDLER_MCTS_RESET = 3
HANDLER_MCTS_STEP  = 4
HANDLER_MCTS_BATCH = 5
FLAG_NONE          = 0
STAGE2_FIXED_EPISODES = 10000
_HDR           = struct.Struct('<Iiii')  # flag, handler_id, query_num, reserved
_LEN           = struct.Struct('<I')     # 4-byte LE uint32 length prefix
HEADER_SIZE    = _HDR.size               # 16 bytes


def _ws(s: str) -> bytes:
    """WriteString: i32 len + utf-8 (PacketWriter.WriteString 동일 형식)."""
    b = s.encode('utf-8')
    return struct.pack('<i', len(b)) + b


def _pack(handler_id: int, body: bytes) -> bytes:
    """app-level 헤더 + TCP length-prefix frame으로 래핑."""
    payload = _HDR.pack(FLAG_NONE, handler_id, 0, 0) + body
    return _LEN.pack(len(payload)) + payload


_I32 = struct.Struct('<i')

# opponent name → C# OpponentType i32
_OPPONENT_TYPE: Dict[str, int] = {
    'external': 0,
    'random':   1,
    'greedy':   2,
    'rule_based': 3,
}

def encode_init(
    p1_deck: str, p2_deck: str, p1_id: str, p2_id: str,
    ai_side: int = -1, opponent_type: int = 0, seed: int = -1,
) -> bytes:
    return _pack(
        HANDLER_INIT,
        _ws(p1_deck) + _ws(p2_deck) + _ws(p1_id) + _ws(p2_id)
        + _I32.pack(ai_side) + _I32.pack(opponent_type) + _I32.pack(seed),
    )


def encode_step(action_uid: str) -> bytes:
    # action_uid는 WriteString 형식 (i32 len + utf-8)
    return _pack(HANDLER_STEP, _ws(action_uid))


def encode_mcts_reset() -> bytes:
    return _pack(HANDLER_MCTS_RESET, b'')


def encode_mcts_step(action_uid: str) -> bytes:
    return _pack(HANDLER_MCTS_STEP, _ws(action_uid))


def encode_mcts_batch(action_indices: Sequence[int]) -> bytes:
    body = struct.pack('<i', len(action_indices))
    for idx in action_indices:
        body += _I32.pack(int(idx))
    return _pack(HANDLER_MCTS_BATCH, body)


async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    return await reader.readexactly(n)


async def recv_packet(reader: asyncio.StreamReader, timeout: Optional[float] = None) -> Tuple[int, bytes]:
    """(handler_id, body) 반환. body는 16-byte 헤더 이후의 페이로드.
    timeout(초)이 지정되면 asyncio.TimeoutError 발생.
    """
    if timeout is not None:
        size_buf = await asyncio.wait_for(_read_exact(reader, 4), timeout=timeout)
        (size,)  = _LEN.unpack(size_buf)
        payload  = await asyncio.wait_for(_read_exact(reader, size), timeout=timeout)
    else:
        size_buf = await _read_exact(reader, 4)
        (size,)  = _LEN.unpack(size_buf)
        payload  = await _read_exact(reader, size)
    _, handler_id, _, _ = _HDR.unpack_from(payload, 0)
    return handler_id, payload[HEADER_SIZE:]


def parse_snapshot(body: bytes) -> Dict[str, Any]:
    """
    SnapshotCodec.Write() 출력을 Python dict로 변환.
    trainer.py의 snapshot dict 형식과 호환.
    """
    offset = 0

    result_byte = body[offset]; offset += 1

    wlen = struct.unpack_from('<i', body, offset)[0]; offset += 4
    winner_id = body[offset:offset + wlen].decode('utf-8'); offset += wlen

    turn = struct.unpack_from('<i', body, offset)[0]; offset += 4

    alen = struct.unpack_from('<i', body, offset)[0]; offset += 4
    active_player = body[offset:offset + alen].decode('utf-8'); offset += alen

    svlen = struct.unpack_from('<i', body, offset)[0]; offset += 4
    state_vector = list(struct.unpack_from(f'<{svlen}f', body, offset))
    offset += svlen * 4

    action_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
    actions: List[Dict[str, str]] = []
    action_feature_vectors: List[List[float]] = []

    for _ in range(action_count):
        # uid: WriteString (i32 len + utf-8)
        uid_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        uid = body[offset:offset + uid_len].decode('utf-8'); offset += uid_len

        # effect_id: 1-byte len + utf-8
        eff_len = body[offset]; offset += 1
        effect_id = body[offset:offset + eff_len].decode('utf-8'); offset += eff_len

        source_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        source = body[offset:offset + source_len].decode('utf-8'); offset += source_len

        target_type_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        target_type = body[offset:offset + target_type_len].decode('utf-8'); offset += target_type_len

        target_guid_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        target_guid = body[offset:offset + target_guid_len].decode('utf-8'); offset += target_guid_len

        target_guid2_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        target_guid2 = body[offset:offset + target_guid2_len].decode('utf-8'); offset += target_guid2_len

        pos_x = struct.unpack_from('<i', body, offset)[0]; offset += 4
        pos_y = struct.unpack_from('<i', body, offset)[0]; offset += 4

        # feature vector
        feat_count = struct.unpack_from('<h', body, offset)[0]; offset += 2
        feat_vec = list(struct.unpack_from(f'<{feat_count}f', body, offset))
        offset += feat_count * 4

        actions.append({
            'uid': uid,
            'effect_id': effect_id,
            'source': source,
            'target_type': target_type,
            'target_guid': target_guid,
            'target_guid2': target_guid2,
            'pos_x': pos_x,
            'pos_y': pos_y,
        })
        action_feature_vectors.append(feat_vec)

    player_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
    players: List[Dict[str, Any]] = []
    for _ in range(player_count):
        pid_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        pid = body[offset:offset + pid_len].decode('utf-8'); offset += pid_len

        hand_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
        deck_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
        trash_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
        hand_size = struct.unpack_from('<i', body, offset)[0]; offset += 4
        hand: List[Dict[str, Any]] = []
        for _ in range(hand_size):
            uid_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
            card_uid = body[offset:offset + uid_len].decode('utf-8'); offset += uid_len
            card_id_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
            card_id = body[offset:offset + card_id_len].decode('utf-8'); offset += card_id_len
            name_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
            name = body[offset:offset + name_len].decode('utf-8'); offset += name_len
            hand.append({
                'uid': card_uid,
                'card_id': card_id,
                'name': name,
            })
        players.append({
            'id': pid,
            'hand_count': hand_count,
            'deck_count': deck_count,
            'trash_count': trash_count,
            'hand': hand,
        })

    board_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
    board: List[Dict[str, Any]] = []
    for _ in range(board_count):
        uid_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        card_uid = body[offset:offset + uid_len].decode('utf-8'); offset += uid_len
        card_id_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        card_id = body[offset:offset + card_id_len].decode('utf-8'); offset += card_id_len
        name_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        name = body[offset:offset + name_len].decode('utf-8'); offset += name_len
        owner_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        owner = body[offset:offset + owner_len].decode('utf-8'); offset += owner_len
        role_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        role = body[offset:offset + role_len].decode('utf-8'); offset += role_len
        atk = struct.unpack_from('<i', body, offset)[0]; offset += 4
        effective_atk = struct.unpack_from('<i', body, offset)[0]; offset += 4
        hp = struct.unpack_from('<i', body, offset)[0]; offset += 4
        max_hp = struct.unpack_from('<i', body, offset)[0]; offset += 4
        is_placed = bool(body[offset]); offset += 1
        is_moved = bool(body[offset]); offset += 1
        is_attacked = bool(body[offset]); offset += 1
        pos_x = struct.unpack_from('<i', body, offset)[0]; offset += 4
        pos_y = struct.unpack_from('<i', body, offset)[0]; offset += 4
        status_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
        statuses: List[Dict[str, Any]] = []
        for _ in range(status_count):
            stype_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
            stype = body[offset:offset + stype_len].decode('utf-8'); offset += stype_len
            sval = struct.unpack_from('<i', body, offset)[0]; offset += 4
            statuses.append({'type': stype, 'value': sval})
        board.append({
            'uid': card_uid,
            'card_id': card_id,
            'name': name,
            'owner': owner,
            'role': role,
            'atk': atk,
            'effective_atk': effective_atk,
            'hp': hp,
            'max_hp': max_hp,
            'is_placed': is_placed,
            'is_moved': is_moved,
            'is_attacked': is_attacked,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'statuses': statuses,
        })

    result_map = {0: 'Ongoing', 1: 'Player1Win', 2: 'Player2Win', 3: 'Draw'}
    return {
        'result': result_map.get(result_byte, 'Ongoing'),
        'winner_id': winner_id,
        'turn': int(turn),
        'active_player': active_player,
        'state_vector': state_vector,
        'action_feature_vectors': action_feature_vectors,
        'actions': actions,
        'players': players,
        'board': board,
    }


def parse_mcts_leaf_batch(body: bytes) -> List[Dict[str, Any]]:
    """
    MctsBatchResultCodec 출력(leaf 최소 포맷)을 Python dict list로 변환.
    result / winner_id / state_vector만 담는다.
    """
    offset = 0
    count = struct.unpack_from('<i', body, offset)[0]
    offset += 4
    snaps: List[Dict[str, Any]] = []

    result_map = {0: 'Ongoing', 1: 'Player1Win', 2: 'Player2Win', 3: 'Draw'}
    for _ in range(count):
        frame_len = struct.unpack_from('<i', body, offset)[0]
        offset += 4
        frame_body = body[offset:offset + frame_len]
        offset += frame_len

        foff = 0
        result_byte = frame_body[foff]; foff += 1

        wlen = struct.unpack_from('<i', frame_body, foff)[0]; foff += 4
        winner_id = frame_body[foff:foff + wlen].decode('utf-8'); foff += wlen

        svlen = struct.unpack_from('<i', frame_body, foff)[0]; foff += 4
        state_vector = list(struct.unpack_from(f'<{svlen}f', frame_body, foff))
        foff += svlen * 4

        snaps.append({
            'result': result_map.get(result_byte, 'Ongoing'),
            'winner_id': winner_id,
            'state_vector': state_vector,
        })

    return snaps


# ---------------------------------------------------------------------------
# Async 환경 — PythonNetSession의 async 대체
# ---------------------------------------------------------------------------

class RLServerEnv:
    """
    RL_Server 포트 1개에 대한 TCP 연결 1개 = 게임 슬롯 1개.
    init_game / apply_action 은 PythonNetSession과 동일한 인터페이스.
    """

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self._reader = reader
        self._writer = writer

    @classmethod
    async def connect(cls, host: str, port: int, timeout: float = 10.0) -> 'RLServerEnv':
        import socket as _socket
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        sock = writer.transport.get_extra_info('socket')
        if sock is not None:
            sock.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
        return cls(reader, writer)

    async def init_game(
        self, *,
        player1_deck: str,
        player2_deck: str,
        player1_id: str = 'P1',
        player2_id: str = 'P2',
        ai_side: int = -1,
        opponent_type: int = 0,
        seed: int = -1,
    ) -> Dict[str, Any]:
        self._writer.write(encode_init(
            player1_deck, player2_deck, player1_id, player2_id,
            ai_side, opponent_type, seed,
        ))
        _, body = await recv_packet(self._reader)
        return parse_snapshot(body)

    async def apply_action(self, action_uid: str) -> Dict[str, Any]:
        self._writer.write(encode_step(action_uid))
        _, body = await recv_packet(self._reader)
        return parse_snapshot(body)

    async def mcts_reset(self, timeout: float = 10.0) -> Dict[str, Any]:
        """live game을 MCTS 복사본으로 포크."""
        self._writer.write(encode_mcts_reset())
        try:
            _, body = await recv_packet(self._reader, timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f'mcts_reset timeout ({timeout}s): C# server did not respond. '
                'Check server logs — MctsReset may have thrown an exception.'
            )
        return parse_snapshot(body)

    async def mcts_step(self, action_uid: str, timeout: float = 10.0) -> Dict[str, Any]:
        """MCTS 복사본에 액션 적용."""
        self._writer.write(encode_mcts_step(action_uid))
        try:
            _, body = await recv_packet(self._reader, timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f'mcts_step({action_uid!r}) timeout ({timeout}s): C# server did not respond. '
                'Check server logs — MctsStep may have thrown an exception (e.g. MctsReset not called first).'
            )
        return parse_snapshot(body)

    async def mcts_batch_step(self, action_indices: Sequence[int], timeout: float = 10.0) -> List[Dict[str, Any]]:
        """live/root 상태에서 여러 후보를 서버가 한 번에 fork 평가.

        서버는 full snapshot 대신 leaf에 필요한 최소 필드만 돌려준다.
        """
        self._writer.write(encode_mcts_batch(action_indices))
        try:
            _, body = await recv_packet(self._reader, timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f'mcts_batch_step timeout ({timeout}s): C# server did not respond.'
            )
        return parse_mcts_leaf_batch(body)

    async def close(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass



# ---------------------------------------------------------------------------
# 배치 레이아웃 (기존 trainer._balanced_match_layout 과 동일)
# ---------------------------------------------------------------------------

_ALL_LAYOUTS = [
    {'player1_is_ai': True,  'ai_deck': 'Orange',    'opp_deck': 'Orange'},
    {'player1_is_ai': True,  'ai_deck': 'Orange',    'opp_deck': 'Charlotte'},
    {'player1_is_ai': True,  'ai_deck': 'Charlotte', 'opp_deck': 'Orange'},
    {'player1_is_ai': True,  'ai_deck': 'Charlotte', 'opp_deck': 'Charlotte'},
    {'player1_is_ai': False, 'ai_deck': 'Orange',    'opp_deck': 'Orange'},
    {'player1_is_ai': False, 'ai_deck': 'Orange',    'opp_deck': 'Charlotte'},
    {'player1_is_ai': False, 'ai_deck': 'Charlotte', 'opp_deck': 'Orange'},
    {'player1_is_ai': False, 'ai_deck': 'Charlotte', 'opp_deck': 'Charlotte'},
]

_LAYOUTS = list(_ALL_LAYOUTS)

_DECKS = {
    'Orange':    json.dumps(['Or_L', 'Or_B', 'Or_N', 'Or_R', 'Or_P', 'Or_P', 'Or_P']),
    'Charlotte': json.dumps(['Cl_L', 'Cl_B', 'Cl_N', 'Cl_R', 'Cl_P', 'Cl_P', 'Cl_P']),
}


def _layout(idx: int) -> Dict[str, Any]:
    return _LAYOUTS[idx % len(_LAYOUTS)]


def _layout_label(idx: int) -> str:
    layout = _layout(idx)
    side = 'P1' if bool(layout['player1_is_ai']) else 'P2'
    return f"{idx % len(_LAYOUTS)}:{side}:{layout['ai_deck']}_vs_{layout['opp_deck']}"


def _normalize_deck_filter(value: str) -> Optional[str]:
    value = str(value or 'both').strip().lower()
    if value in {'both', 'any', 'all'}:
        return None
    mapping = {'orange': 'Orange', 'charlotte': 'Charlotte'}
    if value not in mapping:
        raise ValueError(f'invalid deck filter: {value!r}')
    return mapping[value]


def _set_layout_filter(ai_deck: str = 'both', opp_deck: str = 'both') -> None:
    global _LAYOUTS
    ai_filter = _normalize_deck_filter(ai_deck)
    opp_filter = _normalize_deck_filter(opp_deck)
    filtered = [
        layout for layout in _ALL_LAYOUTS
        if (ai_filter is None or layout['ai_deck'] == ai_filter)
        and (opp_filter is None or layout['opp_deck'] == opp_filter)
    ]
    if not filtered:
        raise ValueError(
            f'no layouts remain after filtering: ai_deck={ai_deck!r} opp_deck={opp_deck!r}'
        )
    _LAYOUTS = filtered


# ---------------------------------------------------------------------------
# _PortWorker: 스레드-로컬 asyncio 루프로 1개 서버 × N 에이전트 관리
# ---------------------------------------------------------------------------

class _PortWorker:
    """
    독립 스레드에서 asyncio 루프를 유지하며 N개 TCP 연결을 지속 관리.
    trainer를 공유하므로 PyTorch가 GIL을 해제하는 구간에서 M 스레드 간 추론이 병렬 실행.
    """

    def __init__(
        self,
        port: int,
        n_agents: int,
        trainer,
        host: str,
        max_turns: int = 100,
        reward_mode: str = 'dense_if_full',
        mcts_depth: int = 0,
        mcts_top_k: int = 2,
    ) -> None:
        self.port        = port
        self.n_agents    = n_agents
        self.trainer     = trainer
        self.host        = host
        self.max_turns   = max_turns
        self.reward_mode = reward_mode
        self.mcts_depth  = max(0, int(mcts_depth))
        self.mcts_top_k  = max(1, int(mcts_top_k))
        self._task_q:   queue.SimpleQueue = queue.SimpleQueue()
        self._result_q: queue.SimpleQueue = queue.SimpleQueue()
        self._ready    = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self) -> None:
        self._ready.wait()

    def submit(self, it: int, base_idx: int, total_envs: int, opponents, layouts=None) -> None:
        """opponents: single agent (broadcast) or list[agent] (one per env slot)."""
        self._task_q.put((it, base_idx, total_envs, opponents, layouts))

    def get_result(self) -> List[Dict[str, Any]]:
        return self._result_q.get()

    def stop(self) -> None:
        self._task_q.put(None)
        self._thread.join()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        except BaseException as exc:
            self._ready.set()
            self._result_q.put([{
                '__worker_error__': True,
                'port': self.port,
                'error': repr(exc),
                'traceback': traceback.format_exc(),
            }])
        finally:
            loop.close()

    async def _async_main(self) -> None:
        tasks = [RLServerEnv.connect(self.host, self.port) for _ in range(self.n_agents)]
        envs  = list(await asyncio.gather(*tasks))
        print(f'[Worker:{self.port}] connected {self.n_agents} agents')
        self._ready.set()

        loop = asyncio.get_running_loop()
        while True:
            task = await loop.run_in_executor(None, self._task_q.get)
            if task is None:
                break
            it, base_idx, total_envs, opponents, layouts = task
            results = await self._collect(envs, it, base_idx, total_envs, opponents, layouts)
            self._result_q.put(results)

        await asyncio.gather(*[e.close() for e in envs])

    async def _collect(
        self,
        envs: List[RLServerEnv],
        it: int,
        base_idx: int,
        total_envs: int,
        opponents,
        layouts=None,
    ) -> List[Dict[str, Any]]:
        from RL_AI.training.storage import RolloutBuffer, RolloutStep
        from RL_AI.training.reward import dense_reward_from_transition, terminal_reward_for_player
        from RL_AI.SeaEngine.action_adapter import choose_action_with_agent
        from RL_AI.agents import SeaEngineRandomAgent

        n = len(envs)
        if not isinstance(opponents, (list, tuple)):
            opponents = [opponents] * n
        if layouts is not None and not isinstance(layouts, (list, tuple)):
            layouts = [layouts] * n
        _fallback = SeaEngineRandomAgent()

        reward_mode = str(getattr(self, 'reward_mode', 'dense_if_full')).strip().lower()

        def _transition_reward(
            *,
            prev_snap: Dict[str, Any],
            new_snap: Dict[str, Any],
            action: Dict[str, Any],
            ai_id: str,
        ) -> float:
            effect_id = str(action.get('effect_id', ''))
            if reward_mode == 'terminal':
                return 0.0
            if reward_mode == 'dense_if_full':
                try:
                    return float(dense_reward_from_transition(
                        prev_snap, new_snap, ai_id=ai_id, action_effect_id=effect_id,
                    ))
                except Exception:
                    pass
            reward = 0.0
            if effect_id == 'DefaultAttack':
                reward += 0.015
            elif effect_id == 'DeployUnit':
                reward += 0.006
            elif effect_id == 'TurnEnd':
                reward -= 0.008
            elif effect_id in {'DefaultMove', 'PawnGeneric'}:
                reward -= 0.002
            return max(-0.03, min(0.03, reward))

        # --- Init all N games in parallel ---
        async def _init(i: int):
            layout_idx  = int(layouts[i]) if layouts is not None and i < len(layouts) else it * total_envs + base_idx + i
            layout      = _layout(layout_idx)
            p1_is_ai    = bool(layout['player1_is_ai'])
            ai_id       = 'P1' if p1_is_ai else 'P2'
            opp_id      = 'P2' if p1_is_ai else 'P1'
            p1_deck     = _DECKS[layout['ai_deck']  if p1_is_ai else layout['opp_deck']]
            p2_deck     = _DECKS[layout['opp_deck'] if p1_is_ai else layout['ai_deck']]
            ai_side     = 0 if p1_is_ai else 1
            opp         = opponents[i] if i < len(opponents) else None
            opp_name    = str(getattr(opp, 'name', '') or '').strip()
            opp_type    = _OPPONENT_TYPE.get(opp_name, 0)
            snap        = await envs[i].init_game(
                player1_deck=p1_deck, player2_deck=p2_deck,
                player1_id='P1', player2_id='P2',
                ai_side=ai_side, opponent_type=opp_type,
            )
            return ai_id, p1_is_ai, layout_idx % len(_LAYOUTS), opp_type, snap

        init_results = await asyncio.gather(*[_init(i) for i in range(n)])
        ai_ids       = [r[0] for r in init_results]
        ai_is_p1     = [bool(r[1]) for r in init_results]
        layout_ids   = [int(r[2]) for r in init_results]
        opp_types    = [r[3] for r in init_results]
        snaps        = [r[4] for r in init_results]
        buffers = [RolloutBuffer() for _ in range(n)]
        steps   = [0] * n
        done    = [False] * n

        # --- Step-sync batched loop ---
        while True:
            active = [
                i for i in range(n)
                if not done[i]
                and snaps[i]['result'] == 'Ongoing'
                and snaps[i]['turn'] <= self.max_turns
                and snaps[i].get('actions')
            ]
            if not active:
                break

            ai_idx  = [i for i in active if snaps[i]['active_player'] == ai_ids[i]]
            opp_idx = [i for i in active
                       if snaps[i]['active_player'] != ai_ids[i] and opp_types[i] == 0]

            chosen: Dict[int, Dict[str, Any]] = {}
            if ai_idx:
                sv   = [snaps[i]['state_vector']           for i in ai_idx]
                av   = [snaps[i]['action_feature_vectors'] for i in ai_idx]
                la   = [snaps[i]['actions']                for i in ai_idx]
                outs = self.trainer.agent.compute_policy_output_batch(sv, av, la)
                for k, i in enumerate(ai_idx):
                    out = outs[k]
                    buffers[i].add_step(RolloutStep(
                        episode_id=it * total_envs + base_idx + i,
                        player_id=0,
                        state_vector=out.state_vector,
                        action_feature_vectors=out.action_feature_vectors,
                        chosen_action_index=out.action_index,
                        reward=0.0,
                        done=False,
                        old_log_prob=out.log_prob,
                        old_value=out.value,
                    ))
                    chosen[i] = out.action

            for i in opp_idx:
                opp = opponents[i] if opponents[i] is not None else _fallback
                _, act = choose_action_with_agent(opp, snaps[i])
                chosen[i] = act

            step_idx = [i for i in active if i in chosen]
            for i in active:
                if i not in chosen:
                    done[i] = True

            prev_snaps = {i: snaps[i] for i in step_idx}
            new_snap_list = await asyncio.gather(
                *[envs[i].apply_action(chosen[i]['uid']) for i in step_idx]
            )

            for i, new_snap in zip(step_idx, new_snap_list):
                if buffers[i].steps:
                    buffers[i].steps[-1].reward += _transition_reward(
                        prev_snap=prev_snaps[i], new_snap=new_snap,
                        action=chosen[i], ai_id=ai_ids[i],
                    )
                snaps[i]  = new_snap
                steps[i] += 1
                if new_snap['result'] != 'Ongoing' or new_snap['turn'] > self.max_turns:
                    done[i] = True

        # --- Terminal rewards + GAE ---
        results = []
        for i in range(n):
            snap      = snaps[i]
            winner_id = snap.get('winner_id', '')
            ai_id     = ai_ids[i]
            if buffers[i].steps:
                terminal_player = 'P1' if ai_is_p1[i] else 'P2'
                terminal_r = terminal_reward_for_player(
                    snap.get('result', 'Ongoing'),
                    terminal_player,
                    final_turn=snap.get('turn', None),
                )
                last = buffers[i].steps[-1]
                last.reward += terminal_r
                last.done    = True
                opp_agent = opponents[i] if i < len(opponents) else None
                opp_name  = str(getattr(opp_agent, 'name', '') or '').lower()
                if winner_id == ai_id:
                    traj_weight = 3.0 if opp_name in {'greedy', 'rule_based'} else 2.0
                elif str(snap.get('result', '')) == 'Draw':
                    traj_weight = 1.15
                elif str(snap.get('result', '')) == 'Ongoing':
                    traj_weight = 0.9
                else:
                    traj_weight = 0.85
                for step in buffers[i].steps:
                    step.sample_weight = traj_weight
            buffers[i].compute_returns_and_advantages(
                self.trainer.config.gamma, self.trainer.config.gae_lambda,
            )
            results.append({
                'buffer':       buffers[i],
                'result':       snap['result'],
                'steps':        steps[i],
                'ai_won':       winner_id == ai_id,
                'final_turn':   snap.get('turn', 0),
                'layout_id':    layout_ids[i],
                'layout_label': _layout_label(layout_ids[i]),
            })
        return results


# ---------------------------------------------------------------------------
# TrainingSession: M개 PortWorker 스레드를 조율해 단일 모델 학습
# ---------------------------------------------------------------------------

class TrainingSession:
    """
    M개 스레드가 각자의 asyncio 루프로 N개 게임을 동시에 수행.
    스레드 간 공유 trainer → 하나의 모델이 M×N 경험으로 학습.

    워크플로우: pre-eval → 커리큘럼 스케줄 학습 (self-play 포함) → post-eval
    """

    def __init__(
        self,
        ports: List[int],
        n_agents: int,
        trainer,
        *,
        device: str = 'cpu',
        host: str = '127.0.0.1',
        log_interval: int = 100,
        save_interval: int = 2500,
        total_episodes: int = 10000,
        seed: Optional[int] = None,
        eval_interval: int = 0,
        n_eval: int = 50,
        checkpoint_n_eval: int = 25,
        max_turns: int = 70,
        reward_mode: str = 'dense_if_full',
        benchmark_mode: bool = False,
        early_stage2: bool = False,
        early_stage2_patience: int = 2,
        early_stage2_min_episode: int = 512,
        early_stage2_min_delta: float = 0.01,
        early_stage2_worst_win_rate: float = 0.80,
        eval_sample_actions: bool = False,
    ):
        self.ports          = ports
        self.n_agents       = n_agents
        self.trainer        = trainer
        self.device         = device
        self.host           = host
        self.log_interval   = log_interval
        self.save_interval  = save_interval
        self.total_episodes = total_episodes
        self.seed           = seed
        self.eval_interval  = eval_interval
        self.n_eval         = n_eval
        self.checkpoint_n_eval = max(0, int(checkpoint_n_eval))
        self.max_turns      = max_turns
        self.reward_mode    = reward_mode
        self.benchmark_mode = bool(benchmark_mode)
        self.early_stage2   = bool(early_stage2)
        self.eval_sample_actions = bool(eval_sample_actions)
        self.early_stage2_patience    = max(1, int(early_stage2_patience))
        self.early_stage2_min_episode = max(0, int(early_stage2_min_episode))
        self.early_stage2_min_delta   = max(0.0, float(early_stage2_min_delta))
        self.early_stage2_worst_win_rate = max(0.0, float(early_stage2_worst_win_rate))
        self._best_eval_score: Optional[float] = None
        self.stage2_triggered = False
        self.stage2_handoff_checkpoint = ""

    def _maybe_trigger_stage2(
        self,
        *,
        episode: int,
        eval_score: float,
        checkpoint_path: Path,
        worst_layout_score: Optional[float] = None,
    ) -> bool:
        """Return True when Stage 1 should hand off to Stage 2."""
        if self.benchmark_mode or not self.early_stage2:
            return False
        if episode < self.early_stage2_min_episode:
            return False

        if worst_layout_score is not None and worst_layout_score >= self.early_stage2_worst_win_rate:
            print(
                f'[*] Early-Stage2 worst-layout stop: ep={episode} '
                f'worst={worst_layout_score:.4f} threshold={self.early_stage2_worst_win_rate:.4f}'
            )
            self.stage2_triggered = True
            self.stage2_handoff_checkpoint = str(checkpoint_path)
            print(
                f'[*] Early-Stage2 triggered at ep={episode}. '
                f'checkpoint={checkpoint_path}'
            )
            return True

        score = float(eval_score)
        if self._best_eval_score is None:
            self._best_eval_score = score
            print(
                f'[*] Early-Stage2 monitor init: ep={episode} '
                f'score={score:.4f}'
            )
            return False
        if score >= self._best_eval_score + self.early_stage2_min_delta:
            print(
                f'[*] Early-Stage2 monitor improve: ep={episode} '
                f'score={score:.4f} best={self._best_eval_score:.4f}'
            )
            self._best_eval_score = score
            return False
        if score < self._best_eval_score:
            print(
                f'[*] Early-Stage2 monitor drop: ep={episode} '
                f'score={score:.4f} best={self._best_eval_score:.4f}'
            )
            self.stage2_triggered = True
            self.stage2_handoff_checkpoint = str(checkpoint_path)
            print(
                f'[*] Early-Stage2 triggered at ep={episode}. '
                f'checkpoint={checkpoint_path}'
            )
            return True
        print(
            f'[*] Early-Stage2 monitor hold: ep={episode} '
            f'score={score:.4f} best={self._best_eval_score:.4f}'
        )
        return False

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _merge_buffers(self, results: List[Dict[str, Any]]):
        from RL_AI.training.storage import RolloutBuffer
        merged = RolloutBuffer()
        for r in results:
            if 'buffer' not in r:
                continue
            for step in r['buffer'].steps:
                merged.add_step(step)
        return merged

    def _stage2_checkpoint_path(self, episodes_done: int) -> Path:
        base = Path(_resolve_project_path(self.output_path)) if self.output_path else Path(_resolve_project_path('models/final_model_stage2.pt'))
        stem = base.stem
        suffix = base.suffix or '.pt'
        if stem.endswith('_stage2'):
            stem = stem[:-7]
        return base.with_name(f'{stem}_stage2_ep{episodes_done}{suffix}')

    # ------------------------------------------------------------------
    # Opponent pool
    # ------------------------------------------------------------------

    def _build_opponent_pool(self) -> Tuple[List, Dict[str, Any]]:
        """Returns (pool_list_for_schedule, pool_dict_by_name)."""
        from RL_AI.agents import SeaEngineRandomAgent, SeaEngineGreedyAgent, SeaEngineRuleBasedAgent
        pool_list: List[Any] = []
        pool_dict: Dict[str, Any] = {}
        for cls in (SeaEngineRandomAgent, SeaEngineGreedyAgent, SeaEngineRuleBasedAgent):
            try:
                agent = cls()
                pool_list.append(agent)
                pool_dict[agent.name] = agent
            except Exception:
                pass
        return pool_list, pool_dict

    def _load_self_play_agent(self, ep: int):
        """Load checkpoint ep as a frozen SeaEngineRLAgent for self-play."""
        import torch
        from RL_AI.agents import SeaEngineRLAgent, infer_hidden_dim_from_state_dict, load_state_dict_flexible
        from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM
        path = Path(_resolve_project_path(f'models/ckpt_ep{ep}.pt'))
        if not path.exists():
            return None
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        hidden_dim = infer_hidden_dim_from_state_dict(state_dict)
        agent = SeaEngineRLAgent(hidden_dim=hidden_dim, device='cpu', sample_actions=True)
        state_dim = getattr(self.trainer.agent, 'state_dim', None) or STATE_VECTOR_DIM
        agent.ensure_model(state_dim)
        load_state_dict_flexible(agent.model, state_dict)
        agent.model.eval()
        agent.name = f'self_ep_{ep}'
        return agent

    def _get_batch_opponents(
        self,
        schedule: List[str],
        ep_start: int,
        n: int,
        pool_dict: Dict[str, Any],
        fallback,
    ) -> List[Any]:
        """Map schedule[ep_start:ep_start+n] to a list of opponent agents.

        self_ep_* 이름이 pool_dict에 없으면 checkpoint 로드를 시도한다.
        로드 실패 시 greedy/rule_based를 번갈아 fallback으로 사용하고 pool_dict에
        기록해 이후 같은 이름에 대해 재시도하지 않는다.
        """
        _alt_fallbacks = [
            pool_dict.get('greedy') or pool_dict.get('rule_based') or fallback,
            pool_dict.get('rule_based') or pool_dict.get('greedy') or fallback,
        ]
        _alt_counter = [0]

        def _next_alt() -> Any:
            agent = _alt_fallbacks[_alt_counter[0] % len(_alt_fallbacks)]
            _alt_counter[0] += 1
            return agent

        result = []
        for j in range(n):
            idx  = ep_start + j
            name = schedule[idx] if idx < len(schedule) else 'random'
            if name not in pool_dict and name.startswith('self_ep_'):
                try:
                    ep_num = int(name.split('_')[-1])
                    agent  = self._load_self_play_agent(ep_num)
                    pool_dict[name] = agent if agent is not None else _next_alt()
                except Exception:
                    pool_dict[name] = _next_alt()
            result.append(pool_dict.get(name, fallback))
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _build_eval_opponents(self) -> Dict[str, Any]:
        from RL_AI.agents import SeaEngineRandomAgent, SeaEngineGreedyAgent, SeaEngineRuleBasedAgent
        agents: Dict[str, Any] = {}
        for cls in (SeaEngineRandomAgent, SeaEngineGreedyAgent, SeaEngineRuleBasedAgent):
            try:
                a = cls()
                agents[a.name] = a
            except Exception:
                pass
        if not agents:
            from RL_AI.agents import SeaEngineRandomAgent
            agents['random'] = SeaEngineRandomAgent()
        return agents

    def _run_eval(
        self,
        workers: List[_PortWorker],
        episode: int,
        n_eval: int,
        label: str = 'eval',
        use_fresh_model: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """기존 워커를 재사용해 활성 layout 각각 n_eval판씩 평가. 모델 업데이트 없음.

        use_fresh_model=True 시 새 초기화 모델로 평가 (pre-eval 베이스라인용).
        평가가 끝나면 원래 trainer.agent로 복원.
        """
        eval_opponents = self._build_eval_opponents()
        total_envs     = len(workers) * self.n_agents
        games_per_layout = max(1, int(n_eval))
        stats: Dict[str, Dict[str, Any]] = {}

        # use_fresh_model: 새 초기화 모델로 교체 후 평가, 이후 원래 모델 복원
        orig_agent = None
        if use_fresh_model:
            from RL_AI.agents import SeaEngineRLAgent
            from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM
            fresh_agent = SeaEngineRLAgent(device=self.device, seed=self.seed)
            fresh_agent.ensure_model(STATE_VECTOR_DIM)
            # 초기화 모델은 random weight이므로 greedy(argmax)가 아닌 stochastic으로 실행
            fresh_agent.sample_actions = True
            orig_agent = self.trainer.agent
            self.trainer.agent = fresh_agent

        prev_sample_actions = getattr(self.trainer.agent, 'sample_actions', None)
        if prev_sample_actions is not None and not use_fresh_model:
            self.trainer.agent.sample_actions = self.eval_sample_actions

        try:
            for opp_name, opp_agent in eval_opponents.items():
                wins = losses = draws = steps_total = 0
                layout_stats: Dict[str, Dict[str, Any]] = {}
                for layout_id in range(len(_LAYOUTS)):
                    lw = ll = ld = lsteps = 0
                    remaining = games_per_layout
                    it = 0
                    while remaining > 0:
                        for i, w in enumerate(workers):
                            w.submit(
                                it, i * self.n_agents, total_envs, opp_agent,
                                layouts=[layout_id] * self.n_agents,
                            )
                        for w in workers:
                            results = w.get_result()
                            _raise_worker_errors(results, f'{label}/{opp_name}/{_layout_label(layout_id)}')
                            take = min(remaining, len(results))
                            for r in results[:take]:
                                lsteps += r.get('steps', 0)
                                if r['ai_won']:
                                    lw += 1
                                elif r['result'] in ('Draw', 'Ongoing'):
                                    ld += 1
                                else:
                                    ll += 1
                            remaining -= take
                        it += 1
                    lname = _layout_label(layout_id)
                    ln = lw + ll + ld
                    layout_stats[lname] = {
                        'wins': lw, 'losses': ll, 'draws': ld,
                        'episodes': ln, 'win_rate': lw / max(1, ln),
                        'avg_steps': lsteps / max(1, ln),
                    }
                    wins += lw; losses += ll; draws += ld; steps_total += lsteps
                n_games = wins + losses + draws
                wr = wins / max(1, n_games)
                stats[opp_name] = {
                    'wins': wins, 'losses': losses, 'draws': draws,
                    'episodes': n_games, 'win_rate': wr,
                    'layout_stats': layout_stats,
                    'games_per_layout': games_per_layout,
                }
                ep_tag = f'ep={episode:6d}  ' if label == 'eval' else f'n={n_games}  '
                print(
                    f'[{label}]  {ep_tag}vs {opp_name}: '
                    f'w/l/d={wins}/{losses}/{draws}  wr={wr:.3f}'
                )
                for lname, row in layout_stats.items():
                    print(
                        f'[{label}]      {opp_name} {lname}: '
                        f'w/l/d={row["wins"]}/{row["losses"]}/{row["draws"]}  '
                        f'wr={row["win_rate"]:.3f}'
                    )
        finally:
            if prev_sample_actions is not None:
                self.trainer.agent.sample_actions = prev_sample_actions
            if orig_agent is not None:
                self.trainer.agent = orig_agent

        return stats

    def _write_checkpoint_report(
        self,
        ep: int,
        eval_stats: Dict[str, Dict[str, Any]],
    ) -> Path:
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path(_resolve_project_path(f'log/train_ckpt_{ep}_{ts}.txt'))
        path.parent.mkdir(exist_ok=True)
        avg_wr = sum(s['win_rate'] for s in eval_stats.values()) / max(1, len(eval_stats))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'checkpoint_ep={ep}\n')
            for opp_name, s in eval_stats.items():
                f.write(
                    f'{opp_name}=wins={s["wins"]} ({100 * s["win_rate"]:.1f}%) | '
                    f'losses={s["losses"]} | draws={s["draws"]}\n'
                )
                for lname, row in dict(s.get('layout_stats', {})).items():
                    f.write(
                        f'  {opp_name}/{lname}=wins={row["wins"]} ({100 * row["win_rate"]:.1f}%) | '
                        f'losses={row["losses"]} | draws={row["draws"]}\n'
                    )
            f.write(f'score={avg_wr:.4f}\n')
        print(f'[eval]  Report: {path}  score={avg_wr:.4f}')
        return path

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        from RL_AI.training.experiment import _build_training_opponent_schedule, _format_plan_counts

        workers = [
            _PortWorker(
                port, self.n_agents, self.trainer, self.host, self.max_turns,
                reward_mode=self.reward_mode,
            )
            for port in self.ports
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.wait_ready()

        total_envs = self.n_agents * len(self.ports)
        print(
            f'[train] M={len(self.ports)}  N={self.n_agents}  '
            f'total_envs={total_envs}  target_episodes={self.total_episodes}'
        )

        # -- Opponent pool -------------------------------------------------
        pool_list, pool_dict = self._build_opponent_pool()
        fallback = pool_dict.get('greedy') or pool_dict.get('rule_based') or pool_dict.get('random') or next(iter(pool_dict.values()))

        # -- Pre-training evaluation ---------------------------------------
        # 항상 새 초기화 모델로 실행 → 체크포인트 로드 여부와 무관하게 베이스라인 고정
        pre_eval: Dict[str, Dict[str, Any]] = {}
        if self.n_eval > 0 and not self.benchmark_mode:
            print('[train] Pre-training evaluation ...')
            pre_eval = self._run_eval(workers, 0, self.n_eval, label='pre-eval', use_fresh_model=True)

        # -- Build per-episode opponent schedule ---------------------------
        schedule, plan_counts = _build_training_opponent_schedule(
            opponent_pool=pool_list,
            train_episodes=self.total_episodes,
            num_envs=total_envs,
            save_interval=self.save_interval,
            seed=self.seed,
        )
        print(f'[train] Opponent plan: {_format_plan_counts(plan_counts)}')

        # -- Training loop -------------------------------------------------
        episodes_done    = 0
        wins             = 0
        total            = 0
        updates          = 0
        last_losses: Dict[str, float] = {}
        checkpoint_reports: List[Dict[str, Any]] = []
        opponent_stats: Counter = Counter()
        result_stats:   Counter = Counter()
        layout_stats:   Counter = Counter()
        steps_total      = 0
        final_turn_total = 0
        latest_checkpoint_path = ""
        t0 = time.time()

        iterations = (self.total_episodes + total_envs - 1) // total_envs

        for it in range(iterations):
            if episodes_done >= self.total_episodes:
                break

            collect_started_at = time.perf_counter()
            for i, w in enumerate(workers):
                ep_start = it * total_envs + i * self.n_agents
                slot_opps = self._get_batch_opponents(schedule, ep_start, self.n_agents, pool_dict, fallback)
                w.submit(it, i * self.n_agents, total_envs, slot_opps)

            all_results: List[Dict[str, Any]] = []
            for w in workers:
                all_results.extend(w.get_result())
            _raise_worker_errors(all_results, 'Stage1 collect')
            collect_elapsed = max(1e-9, time.perf_counter() - collect_started_at)

            update_started_at = time.perf_counter()
            merged      = self._merge_buffers(all_results)
            last_losses = self.trainer.update_from_buffer(merged)
            update_elapsed = max(1e-9, time.perf_counter() - update_started_at)
            updates    += 1

            batch_wins   = sum(1 for r in all_results if r['ai_won'])
            batch_steps  = sum(int(r.get('steps', 0) or 0) for r in all_results)
            batch_turns  = sum(int(r.get('final_turn', 0) or 0) for r in all_results)
            batch_by_result = Counter(str(r.get('result', '')) for r in all_results)
            batch_layouts   = Counter(int(r.get('layout_id', -1)) for r in all_results)
            wins          += batch_wins
            total         += len(all_results)
            episodes_done += len(all_results)
            steps_total   += batch_steps
            final_turn_total += batch_turns
            result_stats.update(batch_by_result)
            layout_stats.update(batch_layouts)

            ep_base = it * total_envs
            for j in range(min(total_envs, len(schedule) - ep_base)):
                opponent_stats[schedule[ep_base + j]] += 1

            if episodes_done % self.log_interval < total_envs:
                elapsed = time.time() - t0
                wr = wins / total if total else 0.0
                avg_steps = steps_total / max(1, total)
                avg_turn  = final_turn_total / max(1, total)
                print(
                    f'[train] ep={episodes_done:6d}/{self.total_episodes}  '
                    f'wr={wr:.3f}  '
                    f'loss_p={last_losses.get("policy_loss", 0):.4f}  '
                    f'loss_v={last_losses.get("value_loss", 0):.4f}  '
                    f'kl={last_losses.get("approx_kl", 0.0):+.4f}  '
                    f'clip={last_losses.get("clip_fraction", 0.0):.3f}  '
                    f'grad_norm={last_losses.get("grad_norm", 0.0):.4f}  '
                    f'ep/s={episodes_done / elapsed:.2f}  '
                    f'collect={collect_elapsed:.1f}s  '
                    f'update={update_elapsed:.2f}s  '
                    f'avg_steps={avg_steps:.1f}  '
                    f'avg_turn={avg_turn:.1f}  '
                    f'draw={result_stats.get("Draw", 0)}  '
                    f'timeout={result_stats.get("Ongoing", 0)}  '
                    f'elapsed={_format_elapsed(elapsed)}'
                )

            do_ckpt = (
                not self.benchmark_mode
                and self.save_interval > 0
                and episodes_done % self.save_interval < total_envs
            )
            do_eval = (
                not self.benchmark_mode
                and self.eval_interval > 0
                and self.n_eval > 0
                and episodes_done % self.eval_interval < total_envs
            )

            if do_ckpt:
                checkpoint_path = self._save_checkpoint(episodes_done)
                latest_checkpoint_path = str(checkpoint_path)
                sp = self._load_self_play_agent(episodes_done)
                if sp is not None:
                    pool_dict[sp.name] = sp
                    print(f'[train] Self-play agent added: {sp.name}')

            if do_ckpt or do_eval:
                eval_n = self.checkpoint_n_eval if do_ckpt else self.n_eval
                eval_stats  = self._run_eval(workers, episodes_done, eval_n)
                report_path = self._write_checkpoint_report(episodes_done, eval_stats)
                avg_score   = sum(s['win_rate'] for s in eval_stats.values()) / max(1, len(eval_stats))
                worst_score = _worst_layout_win_rate(eval_stats)
                checkpoint_reports.append({
                    'ep': episodes_done,
                    'report': str(report_path),
                    'score': avg_score,
                    'worst_score': worst_score,
                })
                if do_ckpt and self._maybe_trigger_stage2(
                    episode=episodes_done,
                    eval_score=avg_score,
                    checkpoint_path=Path(latest_checkpoint_path),
                    worst_layout_score=worst_score,
                ):
                    break

        elapsed = time.time() - t0
        if not self.benchmark_mode:
            self._save_final()
        for w in workers:
            w.stop()

        wr = wins / max(total, 1)
        print(
            f'[train] Done. episodes={episodes_done}  '
            f'win_rate={wr:.3f}  updates={updates}  '
            f'elapsed={_format_elapsed(elapsed)}'
        )
        return {
            'episodes':           episodes_done,
            'wins':               wins,
            'losses':             total - wins,
            'updates':            updates,
            'win_rate':           wr,
            'elapsed':            elapsed,
            'last_losses':        last_losses,
            'avg_steps':          steps_total / max(1, total),
            'avg_final_turn':     final_turn_total / max(1, total),
            'result_stats':       dict(result_stats),
            'layout_stats':       {_layout_label(k): v for k, v in sorted(layout_stats.items()) if k >= 0},
            'checkpoint_reports': checkpoint_reports,
            'pre_eval':           pre_eval,
            'post_eval':          {},
            'opponent_plan':      dict(plan_counts),
            'stage2_triggered':   self.stage2_triggered,
            'stage2_handoff_checkpoint': self.stage2_handoff_checkpoint,
            'benchmark_mode':     self.benchmark_mode,
        }

    def _save_checkpoint(self, ep: int) -> Path:
        import torch
        path = Path(_resolve_project_path(f'models/ckpt_ep{ep}.pt'))
        path.parent.mkdir(exist_ok=True)
        torch.save(self.trainer.agent.model.state_dict(), path)
        print(f'[train] Checkpoint: {path}')
        return path

    def _save_final(self) -> Path:
        import torch
        path = Path(_resolve_project_path('models/final_model.pt'))
        path.parent.mkdir(exist_ok=True)
        torch.save(self.trainer.agent.model.state_dict(), path)
        print(f'[train] Saved: {path}')
        return path


# ---------------------------------------------------------------------------
# Stage 2 — TCP MCTS 학습 워커
# ---------------------------------------------------------------------------

class _MCTSConfig:
    def __init__(self, depth: int, top_k: int) -> None:
        self.depth = depth
        self.top_k = top_k


class _MCTSTrainingPortWorker:
    """
    독립 스레드 asyncio 루프.
    AI 턴: MCTS depth-limited 탐색으로 clone 기반 최선 액션 결정 → LIVE 게임에 적용.
    상대 턴: 서버 auto-opponent 처리.
    PPO 학습용 RolloutBuffer 수집 후 반환.
    """

    def __init__(
        self,
        port: int,
        n_agents: int,
        trainer,
        mcts_cfg: '_MCTSConfig',
        host: str,
        max_turns: int = 100,
        opponent_pool: Optional[List] = None,
        policy_temperature: float = 0.0,
    ) -> None:
        self.port          = port
        self.n_agents      = n_agents
        self.trainer       = trainer
        self.mcts_cfg      = mcts_cfg
        self.host          = host
        self.max_turns     = max_turns
        self.opponent_pool = opponent_pool or []
        self.policy_temperature = max(0.0, float(policy_temperature))
        self._opp_counter  = 0
        self._task_q:   queue.SimpleQueue = queue.SimpleQueue()
        self._result_q: queue.SimpleQueue = queue.SimpleQueue()
        self._ready    = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self)      -> None: self._thread.start()
    def wait_ready(self) -> None: self._ready.wait()
    def stop(self)       -> None: self._task_q.put(None); self._thread.join()

    def submit(self, it: int, base_idx: int, total_envs: int) -> None:
        self._task_q.put((it, base_idx, total_envs))

    def get_result(self) -> List[Dict[str, Any]]:
        return self._result_q.get()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()

    @staticmethod
    def _topk_from_logits(
        logits: Sequence[float],
        *,
        valid_count: int,
        top_k: int,
        temperature: float,
    ) -> List[int]:
        import torch

        valid_count = max(0, int(valid_count))
        if valid_count <= 0:
            return [0]

        top_k = max(1, int(top_k))
        k_eff = min(top_k, valid_count)
        row_logits = torch.as_tensor(list(logits)[:valid_count], dtype=torch.float32)

        if float(temperature) <= 0.0:
            return [int(idx) for idx in torch.topk(row_logits, k=k_eff).indices.detach().cpu().tolist()]

        remaining = row_logits.clone()
        chosen: List[int] = []
        temp = max(1e-6, float(temperature))
        for _ in range(k_eff):
            probs = torch.softmax(remaining / temp, dim=0)
            sample_idx = int(torch.multinomial(probs, 1).item())
            chosen.append(sample_idx)
            remaining[sample_idx] = float('-inf')
        return chosen

    async def _async_main(self) -> None:
        envs = list(await asyncio.gather(
            *[RLServerEnv.connect(self.host, self.port) for _ in range(self.n_agents)]
        ))
        print(f'[Stage2Worker:{self.port}] connected {self.n_agents} agents')
        self._ready.set()

        loop = asyncio.get_running_loop()
        while True:
            task = await loop.run_in_executor(None, self._task_q.get)
            if task is None:
                break
            it, base_idx, total_envs = task
            results = await self._collect(envs, it, base_idx, total_envs)
            self._result_q.put(results)

        await asyncio.gather(*[e.close() for e in envs])

    async def _collect(
        self,
        envs: List[RLServerEnv],
        it: int,
        base_idx: int,
        total_envs: int,
    ) -> List[Dict[str, Any]]:
        import torch
        from torch.distributions import Categorical as _Cat
        from RL_AI.training.storage import RolloutBuffer, RolloutStep
        from RL_AI.training.reward import dense_reward_from_transition
        from RL_AI.agents import SeaEngineRandomAgent

        n      = len(envs)
        depth  = self.mcts_cfg.depth
        top_k  = self.mcts_cfg.top_k

        # 상대 에이전트: opponent_pool에서 번갈아 선택, 없으면 random fallback
        if self.opponent_pool:
            _opp_agents = [
                self.opponent_pool[i % len(self.opponent_pool)]
                for i in range(n)
            ]
        else:
            _opp_agents = [SeaEngineRandomAgent() for _ in range(n)]

        # --- Init all N games in parallel ---
        async def _init(i: int):
            layout_idx = it * total_envs + base_idx + i
            layout   = _layout(layout_idx)
            p1_is_ai = bool(layout['player1_is_ai'])
            ai_id    = 'P1' if p1_is_ai else 'P2'
            p1_deck  = _DECKS[layout['ai_deck']  if p1_is_ai else layout['opp_deck']]
            p2_deck  = _DECKS[layout['opp_deck'] if p1_is_ai else layout['ai_deck']]
            opp      = _opp_agents[i] if i < len(_opp_agents) else None
            opp_name = str(getattr(opp, 'name', '') or '').strip()
            opp_type = _OPPONENT_TYPE.get(opp_name, 0)
            ai_side  = 0 if p1_is_ai else 1
            snap     = await envs[i].init_game(
                player1_deck=p1_deck, player2_deck=p2_deck,
                player1_id='P1', player2_id='P2',
                ai_side=ai_side, opponent_type=opp_type,
            )
            return ai_id, p1_is_ai, layout_idx % len(_LAYOUTS), opp_type, opp_name, snap

        init_results = await asyncio.gather(*[_init(i) for i in range(n)])
        ai_ids     = [r[0] for r in init_results]
        ai_is_p1   = [bool(r[1]) for r in init_results]
        layout_ids = [int(r[2]) for r in init_results]
        opp_types  = [int(r[3]) for r in init_results]
        opp_names  = [str(r[4]) for r in init_results]
        opp_ids    = ['P2' if ai_id == 'P1' else 'P1' for ai_id in ai_ids]
        snaps      = [r[5] for r in init_results]
        buffers = [RolloutBuffer() for _ in range(n)]
        steps   = [0] * n
        done    = [False] * n
        batch_policy_s = 0.0
        batch_mcts_s   = 0.0
        batch_leaf_s   = 0.0
        batch_apply_s  = 0.0
        batch_opp_s    = 0.0

        while True:
            active = [
                i for i in range(n)
                if not done[i]
                and snaps[i]['result'] == 'Ongoing'
                and snaps[i]['turn'] <= self.max_turns
                and snaps[i].get('actions')
            ]
            if not active:
                break

            ai_idx  = [i for i in active if snaps[i]['active_player'] == ai_ids[i]]
            opp_idx = [
                i for i in active
                if snaps[i]['active_player'] != ai_ids[i] and opp_types[i] == 0
            ]
            blocked = [
                i for i in active
                if snaps[i]['active_player'] != ai_ids[i] and opp_types[i] != 0
            ]
            if blocked:
                raise RuntimeError(
                    f"Stage2 server auto-opponent did not advance to AI turn; "
                    f"leftover_turns={blocked}"
                )

            chosen: Dict[int, Dict[str, Any]] = {}

            # --- AI 턴: 정책 추론 + MCTS 탐색 → RolloutBuffer 기록 ---
            if ai_idx:
                t_policy = time.perf_counter()
                sv  = [snaps[i]['state_vector']            for i in ai_idx]
                av  = [snaps[i]['action_feature_vectors']  for i in ai_idx]
                la  = [snaps[i]['actions']                 for i in ai_idx]
                policy_outs = self.trainer.agent.compute_policy_output_batch(sv, av, la)
                policy_topk = [
                    self._topk_from_logits(
                        out.logits,
                        valid_count=len(snaps[i]['actions']),
                        top_k=top_k,
                        temperature=self.policy_temperature,
                    )
                    for out, i in zip(policy_outs, ai_idx)
                ]
                batch_policy_s += time.perf_counter() - t_policy

                # MCTS: 매 턴 live game을 새로 fork해서 후보별 가치 평가.
                # policy logits 상위 후보를 고른 뒤, clone 스냅샷의 actions 인덱스로 접근.
                # uid 발급기 불일치 회피를 위해 clone actions[k].uid를 사용한다.
                # 선택된 인덱스로 live 스냅샷 uid를 참조해 apply_action.
                if depth > 0:
                    candidate_indices = policy_topk
                    cand_vals: List[List[float]] = [
                        [float('-inf')] * len(cands) for cands in candidate_indices
                    ]
                    if depth == 1:
                        batch_window = 16
                        batch_jobs: List[Tuple[int, List[int]]] = []
                        for k, cands in enumerate(candidate_indices):
                            if not cands:
                                continue
                            batch_jobs.append((k, list(cands)))

                        if batch_jobs:
                            t_mcts = time.perf_counter()
                            leaf_ks: List[int] = []
                            leaf_js: List[int] = []
                            leaf_sv_m: List = []
                            for start in range(0, len(batch_jobs), batch_window):
                                window_jobs = batch_jobs[start:start + batch_window]
                                batch_snaps = list(await asyncio.gather(*[
                                    envs[ai_idx[k]].mcts_batch_step(indices)
                                    for k, indices in window_jobs
                                ]))
                                for (k, _indices), cand_snaps in zip(window_jobs, batch_snaps):
                                    for j, snap_k in enumerate(cand_snaps):
                                        result_k = snap_k['result']
                                        i = ai_idx[k]
                                        if result_k != 'Ongoing':
                                            winner = snap_k.get('winner_id', '')
                                            if winner == ai_ids[i]:
                                                cand_vals[k][j] = 1.0
                                            elif winner == opp_ids[i]:
                                                cand_vals[k][j] = -1.0
                                            else:
                                                cand_vals[k][j] = 0.0
                                        else:
                                            leaf_ks.append(k)
                                            leaf_js.append(j)
                                            leaf_sv_m.append(snap_k['state_vector'])
                            batch_mcts_s += time.perf_counter() - t_mcts
                            if leaf_sv_m:
                                t_leaf = time.perf_counter()
                                leaf_vals = self.trainer.agent.compute_value_batch(leaf_sv_m)
                                for k, j, val in zip(leaf_ks, leaf_js, leaf_vals):
                                    cand_vals[k][j] = float(val)
                                batch_leaf_s += time.perf_counter() - t_leaf
                    else:
                        for k_idx in range(top_k):
                            has_k = [k for k, cands in enumerate(candidate_indices) if k_idx < len(cands)]
                            if not has_k:
                                break

                            # Phase 2: 후보 k_idx를 clone에 적용 (fork 다시, 매 후보마다 독립)
                            t_mcts = time.perf_counter()
                            fork_snaps = list(await asyncio.gather(
                                *[envs[ai_idx[k]].mcts_reset() for k in has_k]
                            ))
                            # fork_snaps[j] = has_k[j]번 서브인덱스의 clone 스냅샷
                            # clone actions[candidate_indices[k][k_idx]].uid 사용 → uid 발급기 일치
                            curr = {}
                            step_snaps = list(await asyncio.gather(*[
                                envs[ai_idx[k]].mcts_step(
                                    fork_snaps[j]['actions'][candidate_indices[k][k_idx]]['uid']
                                )
                                for j, k in enumerate(has_k)
                            ]))
                            for j, k in enumerate(has_k):
                                curr[k] = step_snaps[j]

                            # Phase 3: depth-1번 추가 rollout (clone 내부, policy 추론)
                            for _ in range(1, depth):
                                ongoing_k = [
                                    k for k in has_k
                                    if curr[k]['result'] == 'Ongoing' and curr[k].get('actions')
                                ]
                                if not ongoing_k:
                                    break
                                sv_d = [curr[k]['state_vector']           for k in ongoing_k]
                                av_d = [curr[k]['action_feature_vectors'] for k in ongoing_k]
                                la_d = [curr[k]['actions']                for k in ongoing_k]
                                d_outs = self.trainer.agent.compute_policy_output_batch(sv_d, av_d, la_d)
                                new_snaps = list(await asyncio.gather(*[
                                    # clone 스냅샷 actions 인덱스 uid 사용
                                    envs[ai_idx[k]].mcts_step(
                                        curr[k]['actions'][d_outs[j].action_index]['uid']
                                    )
                                    for j, k in enumerate(ongoing_k)
                                ]))
                                for k, ns in zip(ongoing_k, new_snaps):
                                    curr[k] = ns
                            batch_mcts_s += time.perf_counter() - t_mcts

                            # Phase 4: leaf 가치 평가
                            leaf_ks: List[int] = []
                            leaf_sv_m: List = []
                            for k in has_k:
                                snap_k = curr[k]
                                result_k = snap_k['result']
                                i = ai_idx[k]
                                if result_k != 'Ongoing':
                                    winner = snap_k.get('winner_id', '')
                                    if winner == ai_ids[i]:
                                        cand_vals[k][k_idx] = 1.0
                                    elif winner == opp_ids[i]:
                                        cand_vals[k][k_idx] = -1.0
                                    else:
                                        cand_vals[k][k_idx] = 0.0
                                else:
                                    leaf_ks.append(k)
                                    leaf_sv_m.append(snap_k['state_vector'])
                            if leaf_sv_m:
                                t_leaf = time.perf_counter()
                                leaf_vals = self.trainer.agent.compute_value_batch(leaf_sv_m)
                                for k, val in zip(leaf_ks, leaf_vals):
                                    cand_vals[k][k_idx] = float(val)
                                batch_leaf_s += time.perf_counter() - t_leaf

                    # 최선 후보 인덱스 → live 스냅샷 uid 참조
                    best_indices = [
                        (
                            random.choice(best_ties)
                            if (candidate_indices[k] and (best_ties := [
                                candidate_indices[k][j]
                                for j, val in enumerate(cand_vals[k])
                                if val == max(cand_vals[k])
                            ]))
                            else policy_outs[k].action_index
                        )
                        for k in range(len(ai_idx))
                    ]
                else:
                    best_indices = [out.action_index for out in policy_outs]

                t_apply = time.perf_counter()
                for k, i in enumerate(ai_idx):
                    policy_out   = policy_outs[k]
                    actions_list = snaps[i]['actions']
                    best_idx     = best_indices[k]

                    n_act = len(actions_list)
                    logits_slice = policy_out.logits[:n_act]
                    _dist = _Cat(logits=torch.tensor(logits_slice))
                    log_prob = float(_dist.log_prob(torch.tensor(best_idx)).item())

                    buffers[i].add_step(RolloutStep(
                        episode_id=it * total_envs + base_idx + i,
                        player_id=0,
                        state_vector=policy_out.state_vector,
                        action_feature_vectors=policy_out.action_feature_vectors,
                        chosen_action_index=best_idx,
                        reward=0.0,
                        done=False,
                        old_log_prob=log_prob,
                        old_value=policy_out.value,
                    ))
                    chosen[i] = actions_list[best_idx]
                batch_apply_s += time.perf_counter() - t_apply

            if opp_idx:
                from RL_AI.SeaEngine.action_adapter import choose_action_with_agent
                t_opp = time.perf_counter()
                for i in opp_idx:
                    opp = _opp_agents[i] if i < len(_opp_agents) else None
                    if opp is None:
                        raise RuntimeError("Stage2 external opponent missing for self-play slot")
                    _, act = choose_action_with_agent(opp, snaps[i])
                    chosen[i] = act
                batch_opp_s += time.perf_counter() - t_opp

            # --- LIVE 게임에 선택된 액션 일괄 적용 ---
            prev_snaps = {i: snaps[i] for i in active}
            t_apply2 = time.perf_counter()
            new_snap_list = await asyncio.gather(
                *[envs[i].apply_action(chosen[i]['uid']) for i in active]
            )
            batch_apply_s += time.perf_counter() - t_apply2

            for i, new_snap in zip(active, new_snap_list):
                effect_id = str(chosen[i].get('effect_id', ''))
                if buffers[i].steps:
                    buffers[i].steps[-1].reward += dense_reward_from_transition(
                        prev_snaps[i], new_snap, ai_id=ai_ids[i], action_effect_id=effect_id,
                    )
                snaps[i]  = new_snap
                steps[i] += 1
                if new_snap['result'] != 'Ongoing' or new_snap['turn'] > self.max_turns:
                    done[i] = True

        # --- 최종 보상 + GAE ---
        results = []
        for i in range(n):
            snap      = snaps[i]
            winner_id = snap.get('winner_id', '')
            ai_id     = ai_ids[i]
            if buffers[i].steps:
                if winner_id == ai_id:
                    terminal_r = 1.0
                elif winner_id and winner_id not in ('None', ai_id):
                    terminal_r = -1.0
                else:
                    terminal_r = 0.0
                last = buffers[i].steps[-1]
                last.reward += terminal_r
                last.done    = True
            buffers[i].compute_returns_and_advantages(
                self.trainer.config.gamma, self.trainer.config.gae_lambda,
            )
            results.append({
                'buffer':     buffers[i],
                'result':     snap['result'],
                'steps':      steps[i],
                'ai_won':     winner_id == ai_id,
                'final_turn': snap.get('turn', 0),
                'layout_id':  layout_ids[i],
                'layout_label': _layout_label(layout_ids[i]),
                'ai_is_p1':   ai_is_p1[i],
                'opponent_type': opp_types[i],
                'opponent_name': opp_names[i],
            })
        results.append({
            'collect_stats': {
                'policy': batch_policy_s,
                'mcts': batch_mcts_s,
                'leaf': batch_leaf_s,
                'opp': 0.0,
                'apply': batch_apply_s,
            }
        })
        return results

    async def _mcts_batch_decide(
        self,
        envs:         List[RLServerEnv],
        snaps:        List[Dict[str, Any]],
        ai_ids_sub:   List[str],
        opp_ids_sub:  List[str],
        policy_outs,
        depth:        int,
        top_k:        int,
    ) -> List[Dict[str, Any]]:
        """
        N개 게임에 대해 동일 candidate 인덱스를 asyncio.gather로 병렬 탐색.
        clone에서 최선 후보를 평가하고 반환. LIVE 게임은 변경하지 않음.
        """
        n          = len(envs)
        candidates = [snap['actions'][:top_k] for snap in snaps]
        max_k      = max((len(c) for c in candidates), default=0)
        cand_vals  = [[float('-inf')] * len(c) for c in candidates]

        for k_idx in range(max_k):
            has_k = [i for i in range(n) if k_idx < len(candidates[i])]
            if not has_k:
                break

            # Phase 1: LIVE 게임 복사 (clone)
            await asyncio.gather(*[envs[i].mcts_reset() for i in has_k])

            # Phase 2: candidate[k_idx] clone에 적용
            d1_snaps = list(await asyncio.gather(*[
                envs[i].mcts_step(candidates[i][k_idx]['uid']) for i in has_k
            ]))
            curr = dict(zip(has_k, d1_snaps))

            # Phase 3: depth-1번 추가 self-play (clone 내부)
            for _ in range(1, depth):
                ongoing = [
                    i for i in has_k
                    if curr[i]['result'] == 'Ongoing' and curr[i].get('actions')
                ]
                if not ongoing:
                    break
                sv_d = [curr[i]['state_vector']            for i in ongoing]
                av_d = [curr[i]['action_feature_vectors']  for i in ongoing]
                la_d = [curr[i]['actions']                 for i in ongoing]
                d_outs = self.trainer.agent.compute_policy_output_batch(sv_d, av_d, la_d)
                new_snaps = await asyncio.gather(*[
                    envs[i].mcts_step(d_outs[j].action['uid'])
                    for j, i in enumerate(ongoing)
                ])
                for i, ns in zip(ongoing, new_snaps):
                    curr[i] = ns

            # Phase 4: leaf 가치 평가
            leaf_idx: List[int] = []
            leaf_sv:  List      = []
            leaf_av:  List      = []
            leaf_la:  List      = []

            for i in has_k:
                snap   = curr[i]
                result = snap['result']
                if result != 'Ongoing':
                    winner = snap.get('winner_id', '')
                    if winner == ai_ids_sub[i]:
                        cand_vals[i][k_idx] = 1.0
                    elif winner == opp_ids_sub[i]:
                        cand_vals[i][k_idx] = -1.0
                    else:
                        cand_vals[i][k_idx] = 0.0
                else:
                    leaf_idx.append(i)
                    leaf_sv.append(snap['state_vector'])
                    leaf_av.append(snap['action_feature_vectors'])
                    leaf_la.append(snap['actions'])

            if leaf_sv:
                leaf_outs = self.trainer.agent.compute_policy_output_batch(
                    leaf_sv, leaf_av, leaf_la
                )
                for i, out in zip(leaf_idx, leaf_outs):
                    cand_vals[i][k_idx] = float(out.value)

        # 최선 후보 반환
        result_actions = []
        for i in range(n):
            cands = candidates[i]
            vals  = cand_vals[i]
            if not cands:
                result_actions.append(policy_outs[i].action)
            else:
                best_k = max(range(len(cands)), key=lambda k: vals[k])
                result_actions.append(cands[best_k])
        return result_actions


# ---------------------------------------------------------------------------
# MCTSTrainingSession: _MCTSTrainingPortWorker 조율 + PPO 업데이트
# ---------------------------------------------------------------------------

class MCTSTrainingSession:
    """
    M개 스레드가 각자의 asyncio 루프에서 MCTS 탐색 + LIVE 게임 진행.
    각 배치 후 trainer.update_from_buffer()로 PPO 업데이트.
    """

    def __init__(
        self,
        ports: List[int],
        n_agents: int,
        trainer,
        *,
        depth: int = 2,
        top_k: int = 2,
        host: str = '127.0.0.1',
        max_turns: int = 100,
        log_interval: int = 100,
        total_episodes: int = 2500,
        opponent_pool: Optional[List] = None,
        env_counts: Optional[Sequence[int]] = None,
        output_path: Optional[str] = None,
        checkpoint_interval: int = 0,
        checkpoint_eval_matches: int = 25,
        early_stop_worst_win_rate: float = 0.80,
        checkpoint_drop_patience: int = 2,
        policy_temperature: float = 0.0,
        checkpoint_eval_fn: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
    ) -> None:
        self.ports          = ports
        self.n_agents       = n_agents
        self.trainer        = trainer
        self.depth          = depth
        self.top_k          = top_k
        self.host           = host
        self.max_turns      = max_turns
        self.opponent_pool  = opponent_pool or []
        self.log_interval   = log_interval
        self.total_episodes = total_episodes
        self.env_counts     = list(env_counts) if env_counts is not None else None
        self.output_path    = output_path
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self.checkpoint_eval_matches = max(0, int(checkpoint_eval_matches))
        self.early_stop_worst_win_rate = max(0.0, float(early_stop_worst_win_rate))
        self.checkpoint_drop_patience = max(1, int(checkpoint_drop_patience))
        self.policy_temperature = max(0.0, float(policy_temperature))
        self.checkpoint_eval_fn = checkpoint_eval_fn

    def _merge_buffers(self, results: List[Dict[str, Any]]):
        from RL_AI.training.storage import RolloutBuffer
        merged = RolloutBuffer()
        for r in results:
            for step in r['buffer'].steps:
                merged.add_step(step)
        return merged

    def _stage2_checkpoint_path(self, episodes_done: int) -> Path:
        base = Path(_resolve_project_path(self.output_path)) if self.output_path else Path(_resolve_project_path('models/final_model_stage2.pt'))
        stem = base.stem
        suffix = base.suffix or '.pt'
        if stem.endswith('_stage2'):
            stem = stem[:-7]
        return base.with_name(f'{stem}_stage2_ep{episodes_done}{suffix}')

    def run(self) -> Dict[str, Any]:
        mcts_cfg = _MCTSConfig(depth=self.depth, top_k=self.top_k)
        port_counts = self.env_counts if self.env_counts is not None else [self.n_agents] * len(self.ports)
        if len(port_counts) != len(self.ports):
            raise ValueError(
                f'env_counts length mismatch: ports={len(self.ports)} env_counts={len(port_counts)}'
            )
        workers = [
            _MCTSTrainingPortWorker(
                port, count, self.trainer, mcts_cfg,
                self.host, self.max_turns,
                opponent_pool=self.opponent_pool,
                policy_temperature=self.policy_temperature,
            )
            for port, count in zip(self.ports, port_counts)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.wait_ready()

        total_envs = sum(port_counts)
        base_offsets: List[int] = []
        running = 0
        for count in port_counts:
            base_offsets.append(running)
            running += count
        print(
            f'[Stage2] M={len(self.ports)}  env_counts={port_counts}  '
            f'total_envs={total_envs}  depth={self.depth}  top_k={self.top_k}  '
            f'target_episodes={self.total_episodes}'
        )

        episodes_done = 0
        wins          = 0
        total         = 0
        updates       = 0
        steps_total      = 0
        final_turn_total  = 0
        collect_policy_total = 0.0
        collect_mcts_total   = 0.0
        collect_leaf_total   = 0.0
        collect_apply_total  = 0.0
        collect_opp_total    = 0.0
        result_stats: Counter = Counter()
        last_losses: Dict[str, float] = {}
        checkpoint_reports: List[Dict[str, Any]] = []
        t0 = time.time()
        next_checkpoint = max(1, int(self.checkpoint_interval)) if self.checkpoint_interval > 0 else 0
        stage2_stop = False
        best_checkpoint_score: Optional[float] = None
        best_checkpoint_path: str = ''
        prev_checkpoint_score: Optional[float] = None
        checkpoint_drop_streak = 0

        iterations = (self.total_episodes + total_envs - 1) // total_envs

        for it in range(iterations):
            if episodes_done >= self.total_episodes or stage2_stop:
                break

            t_batch = time.time()
            print(f'[Stage2] batch {it+1}/{iterations}  collecting {total_envs} episodes ...')
            for i, w in enumerate(workers):
                w.submit(it, base_offsets[i], total_envs)

            all_results: List[Dict[str, Any]] = []
            for w in workers:
                all_results.extend(w.get_result())
            collect_s = time.time() - t_batch
            rollout_results = [r for r in all_results if 'buffer' in r]
            batch_policy_total = 0.0
            batch_mcts_total   = 0.0
            batch_leaf_total   = 0.0
            batch_apply_total  = 0.0
            batch_opp_total    = 0.0
            for r in all_results:
                stats = r.get('collect_stats')
                if not stats:
                    continue
                batch_policy_total += float(stats.get('policy', 0.0))
                batch_mcts_total += float(stats.get('mcts', 0.0))
                batch_leaf_total += float(stats.get('leaf', 0.0))
                batch_opp_total += float(stats.get('opp', 0.0))
                batch_apply_total += float(stats.get('apply', 0.0))

            collect_policy_total += batch_policy_total
            collect_mcts_total += batch_mcts_total
            collect_leaf_total += batch_leaf_total
            collect_opp_total += batch_opp_total
            collect_apply_total += batch_apply_total

            t_update = time.time()
            merged      = self._merge_buffers(rollout_results)
            last_losses = self.trainer.update_from_buffer(merged)
            updates    += 1
            update_s = time.time() - t_update

            batch_wins     = sum(1 for r in rollout_results if r['ai_won'])
            batch_steps    = sum(int(r.get('steps', 0) or 0) for r in rollout_results)
            batch_turns    = sum(int(r.get('final_turn', 0) or 0) for r in rollout_results)
            batch_by_result = Counter(str(r.get('result', '')) for r in rollout_results)
            wins          += batch_wins
            total         += len(rollout_results)
            episodes_done += len(rollout_results)
            steps_total   += batch_steps
            final_turn_total += batch_turns
            result_stats.update(batch_by_result)

            if episodes_done % self.log_interval < total_envs:
                elapsed = time.time() - t0
                ep_s = episodes_done / elapsed if elapsed > 0 else 0.0
                wr = wins / total if total else 0.0
                avg_steps = steps_total / max(1, total)
                avg_turn = final_turn_total / max(1, total)
                batch_breakdown = (
                    f'policy={batch_policy_total:.1f}s  '
                    f'mcts={batch_mcts_total:.1f}s  '
                    f'leaf={batch_leaf_total:.1f}s  '
                    f'opp={batch_opp_total:.1f}s  '
                    f'apply={batch_apply_total:.1f}s'
                )
                print(
                    f'[Stage2] ep={episodes_done:6d}/{self.total_episodes}  '
                    f'wr={wr:.3f}  '
                    f'loss_p={last_losses.get("policy_loss", 0):.4f}  '
                    f'loss_v={last_losses.get("value_loss", 0):.4f}  '
                    f'kl={last_losses.get("approx_kl", 0.0):+.4f}  '
                    f'clip={last_losses.get("clip_fraction", 0.0):.3f}  '
                    f'grad_norm={last_losses.get("grad_norm", 0.0):.4f}  '
                    f'collect={collect_s:.1f}s  '
                    f'(batch: {batch_breakdown})  '
                    f'update={update_s:.1f}s  '
                    f'ep/s={ep_s:.2f}  '
                    f'avg_steps={avg_steps:.1f}  '
                    f'avg_turn={avg_turn:.1f}  '
                    f'draw={result_stats.get("Draw", 0)}  '
                    f'timeout={result_stats.get("Ongoing", 0)}  '
                    f'elapsed={_format_elapsed(elapsed)}'
                )

            if self.checkpoint_interval > 0:
                while episodes_done >= next_checkpoint and not stage2_stop:
                    elapsed = time.time() - t0
                    ep_s = episodes_done / elapsed if elapsed > 0 else 0.0
                    import torch
                    ckpt_path = self._stage2_checkpoint_path(episodes_done)
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    assert self.trainer.agent.model is not None
                    torch.save(self.trainer.agent.model.state_dict(), ckpt_path)
                    print(f'[Stage2] Checkpoint: {ckpt_path}  ep/s={ep_s:.2f}')

                    ckpt_eval: Dict[str, Dict[str, Any]] = {}
                    ckpt_score = 0.0
                    worst_score = 0.0
                    if self.checkpoint_eval_fn is not None and self.checkpoint_eval_matches > 0:
                        print(f'[Stage2] Checkpoint evaluation ... ep={episodes_done}  ep/s={ep_s:.2f}')
                        ckpt_eval = self.checkpoint_eval_fn()
                        if ckpt_eval:
                            ckpt_score = sum(row['win_rate'] for row in ckpt_eval.values()) / max(1, len(ckpt_eval))
                            worst_score = _worst_layout_win_rate(ckpt_eval)
                    checkpoint_reports.append({
                        'episodes': episodes_done,
                        'checkpoint_path': str(ckpt_path),
                        'score': ckpt_score,
                        'worst_score': worst_score,
                        'eval': ckpt_eval,
                    })
                    if ckpt_eval:
                        print(f'[Stage2] Checkpoint score: {ckpt_score:.4f}  worst={worst_score:.4f}  ep/s={ep_s:.2f}')
                    if ckpt_eval and worst_score >= self.early_stop_worst_win_rate:
                        stage2_stop = True
                        print(
                            f'[Stage2] Early stop: worst layout win rate '
                            f'{worst_score:.4f} >= {self.early_stop_worst_win_rate:.4f}  '
                            f'ep/s={ep_s:.2f}'
                        )
                        break
                    if ckpt_eval:
                        if best_checkpoint_score is None:
                            best_checkpoint_score = ckpt_score
                            best_checkpoint_path = str(ckpt_path)
                            print(
                                f'[Stage2] Checkpoint monitor init: score={ckpt_score:.4f} '
                                f'ep/s={ep_s:.2f}'
                            )
                        elif ckpt_score >= best_checkpoint_score:
                            if ckpt_score > best_checkpoint_score:
                                print(
                                    f'[Stage2] Checkpoint monitor improve: '
                                    f'score={ckpt_score:.4f} best={best_checkpoint_score:.4f} '
                                    f'ep/s={ep_s:.2f}'
                                )
                            best_checkpoint_score = ckpt_score
                            best_checkpoint_path = str(ckpt_path)
                        if prev_checkpoint_score is not None and ckpt_score < prev_checkpoint_score:
                            checkpoint_drop_streak += 1
                        else:
                            checkpoint_drop_streak = 0
                        prev_checkpoint_score = ckpt_score
                        if checkpoint_drop_streak >= self.checkpoint_drop_patience:
                            stage2_stop = True
                            print(
                                f'[Stage2] Early stop: checkpoint score dropped '
                                f'{checkpoint_drop_streak} times in a row '
                                f'score={ckpt_score:.4f} best={best_checkpoint_score:.4f} '
                                f'patience={self.checkpoint_drop_patience} '
                                f'ep/s={ep_s:.2f}'
                            )
                            break
                    next_checkpoint += max(1, int(self.checkpoint_interval))

        elapsed = time.time() - t0
        for w in workers:
            w.stop()

        wr = wins / max(total, 1)
        ep_s = episodes_done / elapsed if elapsed > 0 else 0.0
        avg_steps = steps_total / max(1, total)
        avg_turn = final_turn_total / max(1, total)
        print(
            f'[Stage2] Done. episodes={episodes_done}  '
            f'win_rate={wr:.3f}  updates={updates}  '
            f'avg_steps={avg_steps:.1f}  '
            f'avg_turn={avg_turn:.1f}  '
            f'draw={result_stats.get("Draw", 0)}  '
            f'timeout={result_stats.get("Ongoing", 0)}  '
            f'ep/s={ep_s:.2f}  '
            f'elapsed={_format_elapsed(elapsed)}'
        )
        return {
            'episodes':    episodes_done,
            'wins':        wins,
            'win_rate':    wr,
            'updates':     updates,
            'elapsed':     elapsed,
            'last_losses': last_losses,
            'checkpoint_reports': checkpoint_reports,
            'stage2_stopped_early': stage2_stop,
            'best_checkpoint_score': best_checkpoint_score,
            'best_checkpoint_path': best_checkpoint_path,
        }


# ---------------------------------------------------------------------------
# 학습 진입점
# ---------------------------------------------------------------------------

def _ensure_importable() -> None:
    """RL_AI 패키지가 임포트 가능하도록 sys.path 보정."""
    pkg_parent = str(PROJECT_ROOT.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


def _resolve_project_path(path: str) -> str:
    """Resolve a path relative to the RL_AI project root."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    p = _apply_run_namespace(p)
    return str((PROJECT_ROOT / p).resolve())


def _build_trainer(seed: Optional[int], device: str, model_path: Optional[str] = None):
    """SeaEnginePPOTrainer + agent 생성. model_path 지정 시 체크포인트 로드."""
    import torch
    from RL_AI.agents import SeaEngineRLAgent, infer_hidden_dim_from_state_dict, load_state_dict_flexible
    from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM
    from RL_AI.training.trainer import SeaEnginePPOTrainer

    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    agent = SeaEngineRLAgent(device=device, seed=seed)
    if model_path:
        checkpoint = Path(model_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f'checkpoint not found: {checkpoint}')
        state_dict = torch.load(checkpoint, map_location=agent.device)
        hidden_dim = infer_hidden_dim_from_state_dict(state_dict, fallback=agent.hidden_dim)
        agent.hidden_dim = hidden_dim
        agent.ensure_model(STATE_VECTOR_DIM)
        assert agent.model is not None
        load_state_dict_flexible(agent.model, state_dict)
        agent.model.eval()
    trainer = SeaEnginePPOTrainer(agent)
    return trainer


def _eval_stats_line(stats: Dict[str, Dict[str, Any]], prefix: str) -> str:
    """Format per-opponent eval stats as a single summary line."""
    parts = []
    for opp_name, s in stats.items():
        wr = s.get('win_rate', 0.0)
        parts.append(
            f'{opp_name}='
            f'w={s["wins"]},l={s["losses"]},d={s["draws"]},'
            f'wr={wr:.3f}'
        )
    return f'{prefix}=' + ' | '.join(parts) if parts else f'{prefix}=skipped'


def _worst_layout_win_rate(stats: Dict[str, Dict[str, Any]]) -> float:
    """Return the weakest layout win rate across all opponents / layouts."""
    worst: Optional[float] = None
    for row in stats.values():
        layout_stats = row.get('layout_stats') or {}
        if layout_stats:
            for layout_row in layout_stats.values():
                wr = float(layout_row.get('win_rate', 0.0))
                worst = wr if worst is None else min(worst, wr)
        else:
            wr = float(row.get('win_rate', 0.0))
            worst = wr if worst is None else min(worst, wr)
    return float(worst if worst is not None else 0.0)


def _split_stage2_envs(total_envs: int, ports: Sequence[int]) -> Tuple[List[int], List[int]]:
    usable_ports = list(ports)[: max(1, min(len(ports), max(1, int(total_envs))))]
    total = max(1, int(total_envs))
    base = total // len(usable_ports)
    rem  = total % len(usable_ports)
    counts = [base + (1 if i < rem else 0) for i in range(len(usable_ports))]
    return usable_ports, counts


def _run_tcp_eval(
    *,
    trainer,
    ports: Sequence[int],
    env_counts: Sequence[int],
    host: str,
    max_turns: int,
    n_eval: int,
    label: str,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the current trainer policy through TCP workers without updates.

    n_eval is interpreted as games per each active balanced layout.
    """
    from RL_AI.agents import SeaEngineRandomAgent, SeaEngineGreedyAgent, SeaEngineRuleBasedAgent

    if n_eval <= 0:
        return {}

    eval_opponents = {
        'random':     SeaEngineRandomAgent(),
        'greedy':     SeaEngineGreedyAgent(),
        'rule_based': SeaEngineRuleBasedAgent(),
    }
    workers = [
        # checkpoint 평가는 승패만 필요하므로 dense transition reward 계산을 끈다.
        _PortWorker(port, count, trainer, host, max_turns=max_turns, reward_mode='terminal')
        for port, count in zip(ports, env_counts)
        if count > 0
    ]
    total_envs = sum(env_counts)
    stats: Dict[str, Dict[str, Any]] = {}
    prev_sample_actions = getattr(trainer.agent, 'sample_actions', None)
    if prev_sample_actions is not None:
        trainer.agent.sample_actions = False

    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.wait_ready()

        games_per_layout = max(1, int(n_eval))
        base_offsets: List[int] = []
        running = 0
        for count in env_counts:
            base_offsets.append(running)
            running += count

        for opp_name, opp_agent in eval_opponents.items():
            wins = losses = draws = 0
            layout_stats: Dict[str, Dict[str, Any]] = {}
            for layout_id in range(len(_LAYOUTS)):
                lw = ll = ld = 0
                remaining = games_per_layout
                it = 0
                while remaining > 0:
                    for w_idx, worker in enumerate(workers):
                        worker.submit(
                            it, base_offsets[w_idx], total_envs, opp_agent,
                            layouts=[layout_id] * worker.n_agents,
                        )
                    for worker in workers:
                        results = worker.get_result()
                        _raise_worker_errors(results, f'{label}/{opp_name}/{_layout_label(layout_id)}')
                        take = min(remaining, len(results))
                        for row in results[:take]:
                            if row.get('ai_won'):
                                lw += 1
                            elif str(row.get('result', '')) in {'Draw', 'Ongoing'}:
                                ld += 1
                            else:
                                ll += 1
                        remaining -= take
                    it += 1
                lname = _layout_label(layout_id)
                ln = lw + ll + ld
                layout_stats[lname] = {
                    'wins': lw, 'losses': ll, 'draws': ld,
                    'episodes': ln, 'win_rate': lw / max(1, ln),
                }
                wins += lw; losses += ll; draws += ld
            n_games = wins + losses + draws
            wr = wins / max(1, n_games)
            stats[opp_name] = {
                'wins': wins, 'losses': losses, 'draws': draws,
                'episodes': n_games, 'win_rate': wr,
                'layout_stats': layout_stats, 'games_per_layout': games_per_layout,
            }
            print(f'[{label}]  n={n_games:4d}  vs {opp_name}: w/l/d={wins}/{losses}/{draws}  wr={wr:.3f}')
            for lname, row in layout_stats.items():
                print(
                    f'[{label}]      {opp_name} {lname}: '
                    f'w/l/d={row["wins"]}/{row["losses"]}/{row["draws"]}  '
                    f'wr={row["win_rate"]:.3f}'
                )
    finally:
        if prev_sample_actions is not None:
            trainer.agent.sample_actions = prev_sample_actions
        for worker in workers:
            try:
                worker.stop()
            except Exception:
                pass

    return stats


def run_stage2_refinement(
    *,
    checkpoint_path: str,
    output_path: str,
    episodes: int,
    num_envs: int,
    seed: Optional[int],
    device: str,
    max_turns: int,
    log_interval: int = 100,
    ports: Optional[Sequence[int]] = None,
    host: str = '127.0.0.1',
    mcts_depth: int = 1,
    mcts_top_k: int = 2,
    policy_temperature: float = 0.0,
    early_stage2_worst_win_rate: float = 0.80,
    checkpoint_interval: int = 500,
    checkpoint_eval_matches: int = 25,
    checkpoint_drop_patience: int = 2,
) -> None:
    """Stage 1 checkpoint를 MCTSTrainingSession 기반 stage2로 추가 정제."""
    _ensure_importable()
    import torch
    from RL_AI.training.trainer import PPOConfig

    if episodes <= 0:
        return

    checkpoint = Path(_resolve_project_path(checkpoint_path))
    if not checkpoint.exists():
        raise FileNotFoundError(f'Stage 2 checkpoint not found: {checkpoint}')

    out_path = Path(_resolve_project_path(output_path))
    stage2_ports = list(ports or [9000])
    stage2_ports, env_counts = _split_stage2_envs(num_envs, stage2_ports)
    total_envs = sum(env_counts)
    stage2_n_agents = max(1, max(env_counts) if env_counts else 1)

    print('')
    print('[Stage2] MCTSTrainingSession refinement')
    print(f'[Stage2] checkpoint={checkpoint}')
    print(f'[Stage2] output={out_path}')
    print(f'[Stage2] episodes={episodes}  num_envs={total_envs}  device={device}')
    print(f'[Stage2] ports={stage2_ports}  env_counts={env_counts}  host={host}')
    print(f'[Stage2] mcts_depth={mcts_depth}  mcts_top_k={mcts_top_k}')
    print(f'[Stage2] policy_temperature={policy_temperature}')

    trainer = _build_trainer(seed, device, model_path=str(checkpoint))
    from RL_AI.agents import SeaEngineRLAgent, load_state_dict_flexible
    from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM

    assert trainer.agent.model is not None
    self_opponent = SeaEngineRLAgent(
        hidden_dim=trainer.agent.hidden_dim,
        learning_rate=trainer.agent.learning_rate,
        sample_actions=True,
        device='cpu',
        seed=seed,
    )
    self_state = {k: v.detach().cpu().clone() for k, v in trainer.agent.model.state_dict().items()}
    self_opponent.ensure_model(getattr(trainer.agent, 'state_dim', None) or STATE_VECTOR_DIM)
    assert self_opponent.model is not None
    load_state_dict_flexible(self_opponent.model, self_state)
    self_opponent.model.eval()
    self_opponent.name = 'self'
    stage2_config = PPOConfig(
        learning_rate=5e-5,
        clip_epsilon=0.15,
        entropy_coef=0.015,
        update_epochs=2,
        max_grad_norm=0.7,
        target_kl=0.03,
    )
    trainer.config = stage2_config
    trainer.agent.learning_rate = stage2_config.learning_rate
    if trainer.agent.optimizer is not None:
        for group in trainer.agent.optimizer.param_groups:
            group['lr'] = stage2_config.learning_rate
    print(
        '[Stage2] PPOConfig '
        f'lr={stage2_config.learning_rate} '
        f'clip={stage2_config.clip_epsilon} '
        f'entropy={stage2_config.entropy_coef} '
        f'epochs={stage2_config.update_epochs} '
        f'max_grad_norm={stage2_config.max_grad_norm}'
    )

    from RL_AI.agents import SeaEngineGreedyAgent, SeaEngineRandomAgent, SeaEngineRuleBasedAgent
    opponent_pool = [
        SeaEngineGreedyAgent(seed=None if seed is None else seed + 1),
        SeaEngineRuleBasedAgent(seed=None if seed is None else seed + 2),
        self_opponent,
    ]
    session = MCTSTrainingSession(
        ports=stage2_ports,
        n_agents=stage2_n_agents,
        trainer=trainer,
        depth=mcts_depth,
        top_k=mcts_top_k,
        host=host,
        max_turns=max_turns,
        log_interval=log_interval,
        total_episodes=episodes,
        opponent_pool=opponent_pool,
        env_counts=env_counts,
        output_path=str(out_path),
        checkpoint_interval=checkpoint_interval,
        checkpoint_eval_matches=checkpoint_eval_matches,
        early_stop_worst_win_rate=early_stage2_worst_win_rate,
        checkpoint_drop_patience=checkpoint_drop_patience,
        policy_temperature=policy_temperature,
        checkpoint_eval_fn=lambda: _run_tcp_eval(
            trainer=trainer,
            ports=stage2_ports,
            env_counts=env_counts,
            host=host,
            max_turns=max_turns,
            n_eval=checkpoint_eval_matches,
            label='stage2-ckpt',
        ),
    )
    stage2_result = session.run()
    episodes_done = int(stage2_result.get('episodes', 0))
    win_rate = float(stage2_result.get('win_rate', 0.0))
    if episodes_done <= 0:
        raise RuntimeError(
            'Stage2 finished with 0 successful episodes. '
            'Refined model was not saved. Check server logs.'
        )
    best_checkpoint_path = str(stage2_result.get('best_checkpoint_path', '') or '')
    if best_checkpoint_path:
        best_path = Path(best_checkpoint_path)
        if best_path.exists():
            from RL_AI.agents import load_state_dict_flexible
            state_dict = torch.load(best_path, map_location=trainer.agent.device, weights_only=True)
            assert trainer.agent.model is not None
            load_state_dict_flexible(trainer.agent.model, state_dict)
            print(f'[Stage2] Restored best checkpoint: {best_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert trainer.agent.model is not None
    torch.save(trainer.agent.model.state_dict(), out_path)
    print(f'[Stage2] Saved refined model: {out_path}')
    print(f'[Stage2] Done. episodes={episodes_done}  win_rate={win_rate:.3f}')


def run_training(
    ports: List[int],
    n_agents: int,
    total_episodes: int,
    seed: Optional[int],
    device: str,
    host: str,
    log_interval: int,
    save_interval: int,
    log_file: Optional[str] = None,
    eval_interval: int = 0,
    n_eval: int = 50,
    checkpoint_n_eval: int = 25,
    max_turns: int = 70,
    reward_mode: str = 'dense_if_full',
    benchmark_mode: bool = False,
    early_stage2: bool = False,
    early_stage2_patience: int = 2,
    early_stage2_min_episode: int = 512,
    early_stage2_min_delta: float = 0.01,
    early_stage2_worst_win_rate: float = 0.80,
    eval_sample_actions: bool = False,
) -> str:
    """M×N 에이전트가 하나의 모델을 학습. M 스레드 × N asyncio."""
    _ensure_importable()

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(_resolve_project_path(log_file)) if log_file else Path(_resolve_project_path(f'log/train_{ts}.log'))
    _setup_logger(log_path)

    t_start = time.time()
    m       = len(ports)
    effective_save_interval = 0 if benchmark_mode else save_interval
    effective_eval_interval = 0 if benchmark_mode else eval_interval
    effective_n_eval        = 0 if benchmark_mode else n_eval

    print(f'[*] train_client started  pid={os.getpid()}')
    print(
        f'[*] ports={ports}  M={m}  N={n_agents}  total_episodes={total_episodes}  '
        f'device={device}  seed={seed}  reward_mode={reward_mode}  benchmark={benchmark_mode}'
    )

    trainer = _build_trainer(seed, device)

    session = TrainingSession(
        ports=ports,
        n_agents=n_agents,
        trainer=trainer,
        device=device,
        host=host,
        log_interval=log_interval,
        save_interval=effective_save_interval,
        total_episodes=total_episodes,
        seed=seed,
        eval_interval=effective_eval_interval,
        n_eval=effective_n_eval,
        checkpoint_n_eval=checkpoint_n_eval,
        max_turns=max_turns,
        reward_mode=reward_mode,
        benchmark_mode=benchmark_mode,
        early_stage2=early_stage2,
        early_stage2_patience=early_stage2_patience,
        early_stage2_min_episode=early_stage2_min_episode,
        early_stage2_min_delta=early_stage2_min_delta,
        early_stage2_worst_win_rate=early_stage2_worst_win_rate,
        eval_sample_actions=eval_sample_actions,
    )
    result = session.run()

    summary_path = Path(_resolve_project_path('log/train_summary.txt'))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ll           = result['last_losses']
    last_upd_str = ','.join(f'{k}={v:.4f}' for k, v in ll.items()) if ll else '-'
    ep_s         = result['episodes'] / max(result['elapsed'], 0.001)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('=== train_client Training ===\n')
        f.write(f'ports={",".join(str(p) for p in ports)}\n')
        f.write(f'n_agents={n_agents}\n')
        f.write(f'total_envs={m * n_agents}\n')
        f.write(f'total_episodes={total_episodes}\n')
        f.write(f'device={device}\n')
        f.write(f'seed={seed}\n')
        f.write(
            f'save_interval={effective_save_interval}  '
            f'eval_interval={effective_eval_interval}  '
            f'n_eval={effective_n_eval}  '
            f'checkpoint_n_eval={checkpoint_n_eval}\n'
        )
        f.write(f'reward_mode={reward_mode}\n')
        f.write(f'benchmark_mode={benchmark_mode}\n')
        f.write(f'early_stage2={early_stage2}\n')
        f.write(f'eval_layouts={len(_LAYOUTS)}  eval_games_per_layout={effective_n_eval}\n')
        f.write('\n')
        f.write(_eval_stats_line(result.get('pre_eval', {}), 'before') + '\n')
        f.write('\n')
        f.write(
            f'train=episodes={result["episodes"]} | '
            f'wins={result["wins"]} ({100 * result["win_rate"]:.1f}%) | '
            f'losses={result["losses"]} | '
            f'updates={result["updates"]} | '
            f'avg_steps={float(result.get("avg_steps", 0.0)):.1f} | '
            f'avg_turn={float(result.get("avg_final_turn", 0.0)):.1f} | '
            f'results={result.get("result_stats", {})} | '
            f'layout_stats={result.get("layout_stats", {})} | '
            f'last_update={last_upd_str} | '
            f'avg_speed={ep_s:.2f} ep/s | '
            f'elapsed={_format_elapsed(result["elapsed"])}\n'
        )
        for ckpt in result.get('checkpoint_reports', []):
            f.write(
                f'checkpoint_ep={ckpt["ep"]} | '
                f'report={ckpt["report"]} | '
                f'score={ckpt["score"]:.4f}\n'
            )
        f.write('\n')
        f.write(_eval_stats_line(result.get('post_eval', {}), 'after') + '\n')
        if benchmark_mode:
            f.write('final_model=skipped_in_benchmark\n')
        else:
            f.write('final_model=models/final_model.pt\n')
        if result.get('stage2_triggered'):
            f.write(f'stage2_handoff={result.get("stage2_handoff_checkpoint", "")}\n')
        f.write(f'log={log_path}\n')

    print(f'[*] Summary: {summary_path}')
    print(f'[*] Total elapsed: {_format_elapsed(time.time() - t_start)}')
    if result.get('stage2_triggered') and result.get('stage2_handoff_checkpoint'):
        return _resolve_project_path(str(result['stage2_handoff_checkpoint']))
    if benchmark_mode:
        return ''
    return _resolve_project_path('models/final_model.pt')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Game.RL_Server 기반 병렬 RL 학습 클라이언트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--multi',     action='store_true', help='Multi-server 모드')
    p.add_argument('--m-servers', type=int, default=2, help='Multi 모드 서버 수 M')

    p.add_argument('--port',           type=int, default=9000,         help='단일 서버 포트')
    p.add_argument('--base-port',      type=int, default=9000,         help='Multi 모드 시작 포트')
    p.add_argument('--host',           type=str, default='127.0.0.1')
    p.add_argument('--n-agents',       type=int, default=64,           help='서버 1개당 동시 에이전트 수 N')
    p.add_argument('--total-episodes', type=int, default=1024,         help='Stage 1 총 에피소드 수')
    p.add_argument('--seed',           type=int, default=None)
    p.add_argument('--device',         type=str, default='auto',       help='torch device')
    p.add_argument('--log-interval',   type=int, default=100)
    p.add_argument('--save-interval',  type=int, default=512)
    p.add_argument('--log-file',       type=str, default=None)
    p.add_argument('--eval-interval',  type=int, default=0,
                   help='평가 주기 에피소드 수 (0=평가 안 함)')
    p.add_argument('--n-eval',         type=int, default=50,
                   help='평가당 활성 layout 각각의 에피소드 수')
    p.add_argument('--checkpoint-n-eval', type=int, default=25,
                   help='체크포인트 평가 시 활성 layout 각각의 에피소드 수')
    p.add_argument(
        '--ai-deck',
        type=str,
        default='both',
        choices=('both', 'orange', 'charlotte'),
        help='AI가 사용할 덱 필터 (both=양쪽 덱 모두)',
    )
    p.add_argument(
        '--opp-deck',
        type=str,
        default='both',
        choices=('both', 'orange', 'charlotte'),
        help='상대가 사용할 덱 필터 (both=양쪽 덱 모두)',
    )
    p.add_argument('--max-turns',      type=int, default=70,           help='에피소드당 최대 턴 수')
    p.add_argument(
        '--reward-mode', type=str, default='dense_if_full',
        choices=('terminal', 'terminal_action', 'dense_if_full'),
        help='학습 보상 방식',
    )
    p.add_argument(
        '--benchmark-mode', action='store_true',
        help='pre/post eval, checkpoint eval, final save를 끄고 throughput만 측정',
    )
    p.add_argument(
        '--early-stage2', '--stage2', dest='early_stage2', action='store_true',
        help='Stage 1 평가가 정체되면 Stage 2로 조기 전환',
    )
    p.add_argument('--early-stage2-patience',    type=int,   default=2)
    p.add_argument('--early-stage2-min-episode', type=int,   default=512)
    p.add_argument('--early-stage2-min-delta',   type=float, default=0.01)
    p.add_argument('--early-stage2-worst-win-rate', type=float, default=0.80)
    p.add_argument('--stage2-checkpoint-interval', type=int, default=500)
    p.add_argument('--stage2-checkpoint-eval-matches', type=int, default=25)
    p.add_argument('--stage2-score-drop-patience', type=int, default=2,
                   help='Stage 2 checkpoint score가 연속 몇 번 떨어지면 조기 종료할지')
    p.add_argument('--run-tag', type=str, default=None,
                   help='실험 결과를 분리하기 위한 태그 (log/models 하위 디렉터리 및 zip 이름에 사용)')
    p.add_argument('--archive-results', dest='archive_results', action='store_true',
                   default=True, help='실험 종료 후 log/models를 zip으로 아카이브')
    p.add_argument('--no-archive-results', dest='archive_results', action='store_false',
                   help='실험 종료 후 zip 아카이브를 끔')
    p.add_argument('--eval-sample-actions', action='store_true', default=False,
                   help='eval 시 sample_actions=True (stochastic). 기본값 False (greedy/argmax).')

    p.add_argument('--stage2-episodes',    type=int, default=STAGE2_FIXED_EPISODES,
                   help='Stage 2 에피소드 수')
    p.add_argument('--stage2-now',         action='store_true',
                   help='Stage 1 건너뛰고 Stage 2만 실행')
    p.add_argument('--stage2-num-envs',    type=int, default=32,
                   help='Stage 2 총 동시 env 수')
    p.add_argument('--stage2-model-path',  type=str, default='models/final_model.pt')
    p.add_argument('--stage2-output-path', type=str, default='models/final_model_stage2.pt')
    p.add_argument('--stage2-mcts-depth',  type=int, default=1)
    p.add_argument('--stage2-mcts-top-k',  type=int, default=2)
    p.add_argument('--stage2-policy-temperature', type=float, default=0.20,
                   help='Stage 2 MCTS policy prior temperature. 0=deterministic top-k.')

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_layout_filter(args.ai_deck, args.opp_deck)
    _set_run_context(args.run_tag, args.archive_results)

    ports = (
        [args.base_port + i for i in range(args.m_servers)]
        if args.multi or args.m_servers > 1
        else [args.port]
    )

    if args.stage2_now:
        _ensure_importable()
        run_stage2_refinement(
            checkpoint_path=args.stage2_model_path,
            output_path=args.stage2_output_path,
            episodes=args.stage2_episodes,
            num_envs=args.stage2_num_envs,
            seed=args.seed,
            device=args.device,
            max_turns=args.max_turns,
            log_interval=args.log_interval,
            ports=ports,
            host=args.host,
            mcts_depth=args.stage2_mcts_depth,
            mcts_top_k=args.stage2_mcts_top_k,
            policy_temperature=args.stage2_policy_temperature,
            early_stage2_worst_win_rate=args.early_stage2_worst_win_rate,
            checkpoint_interval=args.stage2_checkpoint_interval,
            checkpoint_eval_matches=args.stage2_checkpoint_eval_matches,
            checkpoint_drop_patience=args.stage2_score_drop_patience,
        )
    else:
        stage2_handoff = run_training(
            ports=ports,
            n_agents=args.n_agents,
            total_episodes=args.total_episodes,
            seed=args.seed,
            device=args.device,
            host=args.host,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            log_file=args.log_file,
            eval_interval=args.eval_interval,
            n_eval=args.n_eval,
            checkpoint_n_eval=args.checkpoint_n_eval,
            max_turns=args.max_turns,
            reward_mode=args.reward_mode,
            benchmark_mode=args.benchmark_mode,
            early_stage2=args.early_stage2,
            early_stage2_patience=args.early_stage2_patience,
            early_stage2_min_episode=args.early_stage2_min_episode,
            early_stage2_min_delta=args.early_stage2_min_delta,
            early_stage2_worst_win_rate=args.early_stage2_worst_win_rate,
            eval_sample_actions=args.eval_sample_actions,
        )

        if not args.benchmark_mode:
            stage2_checkpoint = stage2_handoff or args.stage2_model_path
            if args.stage2_model_path and args.stage2_model_path != 'models/final_model.pt':
                stage2_checkpoint = args.stage2_model_path
            run_stage2_refinement(
                checkpoint_path=stage2_checkpoint,
                output_path=args.stage2_output_path,
                episodes=args.stage2_episodes,
                num_envs=args.stage2_num_envs,
                seed=args.seed,
                device=args.device,
                max_turns=args.max_turns,
                log_interval=args.log_interval,
                ports=ports,
                host=args.host,
                mcts_depth=args.stage2_mcts_depth,
                mcts_top_k=args.stage2_mcts_top_k,
                policy_temperature=args.stage2_policy_temperature,
                early_stage2_worst_win_rate=args.early_stage2_worst_win_rate,
                checkpoint_interval=args.stage2_checkpoint_interval,
                checkpoint_eval_matches=args.stage2_checkpoint_eval_matches,
                checkpoint_drop_patience=args.stage2_score_drop_patience,
            )

    _archive_run_artifacts()


if __name__ == '__main__':
    print("Train V: Dev")
    main()
