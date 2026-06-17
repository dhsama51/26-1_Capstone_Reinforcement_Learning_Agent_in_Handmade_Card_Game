"""
mcts_session_client.py — MCTS 탐색 병렬 평가 클라이언트

Game.RL_Server의 HANDLER_MCTS_RESET(3) / HANDLER_MCTS_STEP(4) 핸들러를 사용.
self 에이전트 턴에서 depth-limited value search를 수행.

프로토콜:
  HANDLER_MCTS_RESET (3): body 없음 → live game 복사 후 스냅샷 반환
  HANDLER_MCTS_STEP  (4): WriteString(action_uid) → MCTS 복사본에 적용 후 스냅샷 반환
  응답 포맷은 HANDLER_STEP(2)과 완전히 동일 (parse_snapshot 재사용)

MCTS 탐색 흐름 (depth=D, top_k=K):
  각 self 차례마다:
    candidate k in 0..K-1:
      1. mcts_reset()              ← live game 복사
      2. mcts_step(candidate[k])   ← depth-1: self 이동
      3. mcts_step(best_response)  ← depth-2: 상대 이동 (같은 모델로 추론)
         ...반복 (depth-1번)...
      4. RL value 추론 → candidate_values[k]
    best_action = argmax(candidate_values)

  - 모든 에이전트(connId)에 대해 동일 candidate index를 asyncio.gather로 병렬 처리
  - leaf value 추론은 전체 에이전트를 묶어 compute_policy_output_batch 1회 호출

사용법:
  # RL self-play (MCTS depth=1, top_k=2)
  python mcts_session_client.py --model-path models/final_model.pt \\
    --depth 1 --top-k 2 --multi --m-servers 4 --total-matches 400

  # RL vs rule_based (MCTS로 RL 강화)
  python mcts_session_client.py --self rl --model-path models/final.pt \\
    --opp rule_based --depth 1 --top-k 2 --total-matches 400

  # depth=0: MCTS 없이 순수 policy (eval_client.py와 동등)
  python mcts_session_client.py --model-path models/final.pt --depth 0
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import queue
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def _setup_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.stdout, f)   # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, f)   # type: ignore[assignment]
    print(f'[*] log file: {log_file}')


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d} ({seconds:.1f}s)'


# ---------------------------------------------------------------------------
# Binary Protocol
# ---------------------------------------------------------------------------

HANDLER_INIT       = 1
HANDLER_STEP       = 2
HANDLER_MCTS_RESET = 3
HANDLER_MCTS_STEP  = 4
FLAG_NONE          = 0
_HDR               = struct.Struct('<Iiii')
_LEN               = struct.Struct('<I')
HEADER_SIZE        = _HDR.size   # 16 bytes


def _ws(s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<i', len(b)) + b


def _pack(handler_id: int, body: bytes) -> bytes:
    payload = _HDR.pack(FLAG_NONE, handler_id, 0, 0) + body
    return _LEN.pack(len(payload)) + payload


def encode_init(p1_deck: str, p2_deck: str, p1_id: str, p2_id: str) -> bytes:
    return _pack(HANDLER_INIT, _ws(p1_deck) + _ws(p2_deck) + _ws(p1_id) + _ws(p2_id))


def encode_step(action_uid: str) -> bytes:
    return _pack(HANDLER_STEP, _ws(action_uid))


def encode_mcts_reset() -> bytes:
    return _pack(HANDLER_MCTS_RESET, b'')


def encode_mcts_step(action_uid: str) -> bytes:
    return _pack(HANDLER_MCTS_STEP, _ws(action_uid))


async def _read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    return await reader.readexactly(n)


async def recv_packet(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    size_buf = await _read_exact(reader, 4)
    (size,)  = _LEN.unpack(size_buf)
    payload  = await _read_exact(reader, size)
    _, handler_id, _, _ = _HDR.unpack_from(payload, 0)
    return handler_id, payload[HEADER_SIZE:]


def parse_snapshot(body: bytes) -> Dict[str, Any]:
    offset = 0
    result_byte = body[offset]; offset += 1
    wlen = struct.unpack_from('<i', body, offset)[0]; offset += 4
    winner_id = body[offset:offset + wlen].decode('utf-8'); offset += wlen
    turn = struct.unpack_from('<h', body, offset)[0]; offset += 2
    alen = struct.unpack_from('<i', body, offset)[0]; offset += 4
    active_player = body[offset:offset + alen].decode('utf-8'); offset += alen
    svlen = struct.unpack_from('<i', body, offset)[0]; offset += 4
    state_vector = list(struct.unpack_from(f'<{svlen}f', body, offset))
    offset += svlen * 4
    action_count = struct.unpack_from('<i', body, offset)[0]; offset += 4
    actions: List[Dict[str, str]] = []
    action_feature_vectors: List[List[float]] = []
    for _ in range(action_count):
        uid_len = struct.unpack_from('<i', body, offset)[0]; offset += 4
        uid = body[offset:offset + uid_len].decode('utf-8'); offset += uid_len
        eff_len = body[offset]; offset += 1
        effect_id = body[offset:offset + eff_len].decode('utf-8'); offset += eff_len
        feat_count = struct.unpack_from('<h', body, offset)[0]; offset += 2
        feat_vec = list(struct.unpack_from(f'<{feat_count}f', body, offset))
        offset += feat_count * 4
        actions.append({'uid': uid, 'effect_id': effect_id})
        action_feature_vectors.append(feat_vec)
    result_map = {0: 'Ongoing', 1: 'Player1Win', 2: 'Player2Win', 3: 'Draw'}
    return {
        'result':                result_map.get(result_byte, 'Ongoing'),
        'winner_id':             winner_id,
        'turn':                  int(turn),
        'active_player':         active_player,
        'state_vector':          state_vector,
        'action_feature_vectors': action_feature_vectors,
        'actions':               actions,
    }


# ---------------------------------------------------------------------------
# RLServerEnv — MCTS 메서드 추가
# ---------------------------------------------------------------------------

class RLServerEnv:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
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

    async def init_game(self, *, player1_deck: str, player2_deck: str,
                        player1_id: str = 'P1', player2_id: str = 'P2') -> Dict[str, Any]:
        self._writer.write(encode_init(player1_deck, player2_deck, player1_id, player2_id))
        _, body = await recv_packet(self._reader)
        return parse_snapshot(body)

    async def apply_action(self, action_uid: str) -> Dict[str, Any]:
        self._writer.write(encode_step(action_uid))
        _, body = await recv_packet(self._reader)
        return parse_snapshot(body)

    async def mcts_reset(self) -> Dict[str, Any]:
        """live game을 MCTS 복사본으로 포크. 반환값은 현재 live 상태 스냅샷."""
        self._writer.write(encode_mcts_reset())
        _, body = await recv_packet(self._reader)
        return parse_snapshot(body)

    async def mcts_step(self, action_uid: str) -> Dict[str, Any]:
        """MCTS 복사본에 액션 적용. 형식은 apply_action과 동일."""
        self._writer.write(encode_mcts_step(action_uid))
        _, body = await recv_packet(self._reader)
        return parse_snapshot(body)

    async def close(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 덱 / 시나리오 정의
# ---------------------------------------------------------------------------

_DECKS: Dict[str, str] = {
    '귤':       json.dumps(['Or_L', 'Or_B', 'Or_N', 'Or_R', 'Or_P', 'Or_P', 'Or_P']),
    '샤를로테':  json.dumps(['Cl_L', 'Cl_B', 'Cl_N', 'Cl_R', 'Cl_P', 'Cl_P', 'Cl_P']),
}


def _scenario_definitions(prefix: str) -> List[Dict[str, Any]]:
    deck_pairs = [('귤', '샤를로테'), ('샤를로테', '귤')]
    scenarios: List[Dict[str, Any]] = []
    for self_deck_name, other_deck_name in deck_pairs:
        for side_name, self_is_p1 in [('선공', True), ('후공', False)]:
            for relation_name, use_same in [('같은 덱', True), ('다른 덱', False)]:
                opp_deck_name = self_deck_name if use_same else other_deck_name
                p1_deck = _DECKS[self_deck_name if self_is_p1 else opp_deck_name]
                p2_deck = _DECKS[opp_deck_name if self_is_p1 else self_deck_name]
                scenarios.append({
                    'label':          f'{prefix}/{self_deck_name}/{side_name}/{relation_name}',
                    'self_is_p1':     self_is_p1,
                    'self_deck_name': self_deck_name,
                    'opp_deck_name':  opp_deck_name,
                    'side_name':      side_name,
                    'relation_name':  relation_name,
                    'p1_deck':        p1_deck,
                    'p2_deck':        p2_deck,
                    'self_id':        'Self',
                    'opp_id':         'Opp',
                    'p1_id':          'Self' if self_is_p1 else 'Opp',
                    'p2_id':          'Opp'  if self_is_p1 else 'Self',
                })
    return scenarios


# ---------------------------------------------------------------------------
# 에이전트 팩토리
# ---------------------------------------------------------------------------

def _ensure_importable() -> None:
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


def _build_rl_agent(model_path: str, device: str, seed: Optional[int] = None):
    import torch
    from RL_AI.agents import SeaEngineRLAgent, infer_hidden_dim_from_state_dict, load_state_dict_flexible
    from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    hidden_dim = infer_hidden_dim_from_state_dict(state_dict)
    agent = SeaEngineRLAgent(hidden_dim=hidden_dim, device=device,
                             sample_actions=False, seed=seed)
    agent.ensure_model(STATE_VECTOR_DIM)
    assert agent.model is not None
    load_state_dict_flexible(agent.model, state_dict)
    agent.model.eval()
    agent.name = f'rl({Path(model_path).stem})'
    return agent


def _build_agent(kind: str, *, model_path: Optional[str] = None,
                 device: str = 'cpu', seed: Optional[int] = None,
                 self_agent=None):
    kind = kind.lower().strip()
    if kind == 'self':
        if self_agent is None:
            raise ValueError('--opp self requires self_agent to be built first')
        import copy
        clone = copy.deepcopy(self_agent)
        clone.name = f'{self_agent.name}(clone)'
        return clone
    if kind == 'rl':
        if not model_path:
            raise ValueError('--opp rl requires --opp-model-path')
        return _build_rl_agent(model_path, device, seed)
    _ensure_importable()
    from RL_AI.agents import SeaEngineRandomAgent, SeaEngineGreedyAgent, SeaEngineRuleBasedAgent
    if kind == 'random':
        return SeaEngineRandomAgent(seed=seed)
    if kind == 'greedy':
        return SeaEngineGreedyAgent(seed=seed)
    if kind in ('rule_based', 'rule-based', 'rule'):
        return SeaEngineRuleBasedAgent(seed=seed)
    raise ValueError(f'Unknown agent kind: {kind!r}')


def _is_rl_agent(agent) -> bool:
    return hasattr(agent, 'compute_policy_output_batch')


# ---------------------------------------------------------------------------
# MCTS 설정
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    depth: int    # lookahead 깊이. 0 = MCTS 없음 (pure policy)
    top_k: int    # 탐색할 후보 액션 수


# ---------------------------------------------------------------------------
# _MCTSPortWorker
# ---------------------------------------------------------------------------

class _MCTSPortWorker:
    """
    독립 스레드에서 asyncio 루프 유지.
    self 에이전트 턴: MCTS depth-limited value search (RL only).
    opp 에이전트 턴: 배치 추론 또는 순차 휴리스틱.
    """

    def __init__(self, port: int, n_agents: int, self_agent, opp_agent,
                 mcts_cfg: MCTSConfig, host: str, max_turns: int = 100) -> None:
        self.port       = port
        self.n_agents   = n_agents
        self.self_agent = self_agent
        self.opp_agent  = opp_agent
        self.mcts_cfg   = mcts_cfg
        self.host       = host
        self.max_turns  = max_turns
        self._task_q:   queue.SimpleQueue = queue.SimpleQueue()
        self._result_q: queue.SimpleQueue = queue.SimpleQueue()
        self._ready     = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self)                  -> None: self._thread.start()
    def wait_ready(self)             -> None: self._ready.wait()
    def submit(self, a)              -> None: self._task_q.put(a)
    def get_result(self)             -> List: return self._result_q.get()
    def stop(self)                   -> None: self._task_q.put(None); self._thread.join()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        finally:
            loop.close()

    async def _async_main(self) -> None:
        envs = list(await asyncio.gather(
            *[RLServerEnv.connect(self.host, self.port) for _ in range(self.n_agents)]
        ))
        print(f'[MCTSWorker:{self.port}] connected {self.n_agents} slots')
        self._ready.set()

        loop = asyncio.get_running_loop()
        while True:
            assignments = await loop.run_in_executor(None, self._task_q.get)
            if assignments is None:
                break
            results = await self._collect(envs, assignments)
            self._result_q.put(results)

        await asyncio.gather(*[e.close() for e in envs])

    # ------------------------------------------------------------------
    # 메인 에피소드 루프
    # ------------------------------------------------------------------

    async def _collect(self, envs: List[RLServerEnv],
                       assignments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n          = len(envs)
        self_ids   = [a['self_id'] for a in assignments]   # 항상 'Self'
        opp_ids    = [a['opp_id']  for a in assignments]   # 항상 'Opp'
        use_mcts   = _is_rl_agent(self.self_agent) and self.mcts_cfg.depth > 0
        opp_is_rl  = _is_rl_agent(self.opp_agent)

        snaps = list(await asyncio.gather(*[
            envs[i].init_game(
                player1_deck=assignments[i]['p1_deck'],
                player2_deck=assignments[i]['p2_deck'],
                player1_id=assignments[i]['p1_id'],
                player2_id=assignments[i]['p2_id'],
            )
            for i in range(n)
        ]))
        steps = [0] * n
        done  = [False] * n

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

            chosen: Dict[int, Dict[str, Any]] = {}

            self_turn = [i for i in active if snaps[i]['active_player'] == self_ids[i]]
            opp_turn  = [i for i in active if snaps[i]['active_player'] == opp_ids[i]]

            # --- self 에이전트 ---
            if self_turn:
                if use_mcts:
                    mcts_actions = await self._mcts_batch_decide(
                        [envs[i] for i in self_turn],
                        [snaps[i] for i in self_turn],
                        [self_ids[i] for i in self_turn],
                        [opp_ids[i]  for i in self_turn],
                    )
                    for i, act in zip(self_turn, mcts_actions):
                        chosen[i] = act
                elif _is_rl_agent(self.self_agent):
                    sv  = [snaps[i]['state_vector']            for i in self_turn]
                    av  = [snaps[i]['action_feature_vectors']  for i in self_turn]
                    la  = [snaps[i]['actions']                 for i in self_turn]
                    outs = self.self_agent.compute_policy_output_batch(sv, av, la)
                    for k, i in enumerate(self_turn):
                        chosen[i] = outs[k].action
                else:
                    for i in self_turn:
                        _, act = self.self_agent.select_action(snaps[i], snaps[i]['actions'])
                        chosen[i] = act

            # --- opp 에이전트 ---
            if opp_turn:
                if opp_is_rl:
                    sv  = [snaps[i]['state_vector']            for i in opp_turn]
                    av  = [snaps[i]['action_feature_vectors']  for i in opp_turn]
                    la  = [snaps[i]['actions']                 for i in opp_turn]
                    outs = self.opp_agent.compute_policy_output_batch(sv, av, la)
                    for k, i in enumerate(opp_turn):
                        chosen[i] = outs[k].action
                else:
                    for i in opp_turn:
                        _, act = self.opp_agent.select_action(snaps[i], snaps[i]['actions'])
                        chosen[i] = act

            new_snaps = await asyncio.gather(
                *[envs[i].apply_action(chosen[i]['uid']) for i in active]
            )
            for i, snap in zip(active, new_snaps):
                snaps[i]  = snap
                steps[i] += 1
                if snap['result'] != 'Ongoing' or snap['turn'] > self.max_turns:
                    done[i] = True

        results = []
        for i in range(n):
            snap   = snaps[i]
            winner = snap.get('winner_id', '')
            results.append({
                'scenario_idx':   assignments[i]['scenario_idx'],
                'scenario_label': assignments[i]['label'],
                'side_name':      assignments[i]['side_name'],
                'self_deck_name': assignments[i]['self_deck_name'],
                'opp_deck_name':  assignments[i]['opp_deck_name'],
                'relation_name':  assignments[i]['relation_name'],
                'self_won':       winner == self_ids[i],
                'opp_won':        winner == opp_ids[i],
                'draw':           winner not in (self_ids[i], opp_ids[i]),
                'steps':          steps[i],
                'final_turn':     snap.get('turn', 0),
            })
        return results

    # ------------------------------------------------------------------
    # 배치 MCTS 의사결정
    # ------------------------------------------------------------------

    async def _mcts_batch_decide(
        self,
        envs:     List[RLServerEnv],
        snaps:    List[Dict[str, Any]],
        self_ids: List[str],
        opp_ids:  List[str],
    ) -> List[Dict[str, Any]]:
        """
        N개 게임에 대해 동일 candidate 인덱스를 묶어 asyncio.gather로 병렬 탐색.

        각 candidate 라운드:
          1. 전체 mcts_reset (parallel)
          2. 전체 mcts_step(candidate[k]) (parallel)
          3. depth-1번 mcts_step(best_response) (parallel, 배치 추론)
          4. leaf: batch RL value 추론

        반환: agent별 best action dict.
        """
        n          = len(envs)
        depth      = self.mcts_cfg.depth
        top_k      = self.mcts_cfg.top_k

        # 정책 추론 (fallback + 후보 목록 결정)
        sv = [s['state_vector']            for s in snaps]
        av = [s['action_feature_vectors']  for s in snaps]
        la = [s['actions']                 for s in snaps]
        policy_outs = self.self_agent.compute_policy_output_batch(sv, av, la)

        candidates = []
        candidate_indices = []
        for snap, out in zip(snaps, policy_outs):
            actions = snap['actions']
            ranked = sorted(
                range(len(actions)),
                key=lambda idx: float(out.probabilities[idx]) if idx < len(out.probabilities) else 0.0,
                reverse=True,
            )
            picked = ranked[:top_k]
            candidate_indices.append(picked)
            candidates.append([actions[idx] for idx in picked])
        max_k = max((len(c) for c in candidates), default=0)

        # candidate_values[i][k]: agent i, candidate k에 대한 value 추정치
        cand_values = [[float('-inf')] * len(c) for c in candidates]

        for k_idx in range(max_k):
            has_k = [i for i in range(n) if k_idx < len(candidates[i])]
            if not has_k:
                break

            # Phase 1: 해당 agent 전체 mcts_reset
            reset_snaps = list(await asyncio.gather(*[envs[i].mcts_reset() for i in has_k]))
            reset_by_idx = dict(zip(has_k, reset_snaps))

            # Phase 2: candidate k_idx 적용
            step_uids: Dict[int, str] = {}
            for i in has_k:
                live_uid = str(candidates[i][k_idx].get('uid', ''))
                reset_actions = reset_by_idx[i].get('actions') or []
                mapped = next(
                    (a for a in reset_actions if str(a.get('uid', '')) == live_uid),
                    None,
                )
                if mapped is None and reset_actions:
                    original_idx = candidate_indices[i][k_idx]
                    if 0 <= original_idx < len(reset_actions):
                        mapped = reset_actions[original_idx]
                    else:
                        mapped = reset_actions[min(k_idx, len(reset_actions) - 1)]
                if mapped is not None:
                    step_uids[i] = str(mapped.get('uid', ''))

            has_step = [i for i in has_k if step_uids.get(i)]
            if not has_step:
                continue
            d1_snaps = list(await asyncio.gather(*[
                envs[i].mcts_step(step_uids[i]) for i in has_step
            ]))
            curr = dict(zip(has_step, d1_snaps))

            # Phase 3: depth-1번 추가 스텝 (같은 모델로 self-play)
            for _d in range(1, depth):
                ongoing = [i for i in has_step
                           if curr[i]['result'] == 'Ongoing' and curr[i].get('actions')]
                if not ongoing:
                    break

                sv_d = [curr[i]['state_vector']            for i in ongoing]
                av_d = [curr[i]['action_feature_vectors']  for i in ongoing]
                la_d = [curr[i]['actions']                 for i in ongoing]
                d_outs = self.self_agent.compute_policy_output_batch(sv_d, av_d, la_d)

                new_snaps = await asyncio.gather(*[
                    envs[i].mcts_step(d_outs[j].action['uid'])
                    for j, i in enumerate(ongoing)
                ])
                for i, ns in zip(ongoing, new_snaps):
                    curr[i] = ns

            # Phase 4: leaf value 평가
            leaf_idx: List[int] = []
            leaf_sv:  List = []
            leaf_av:  List = []
            leaf_la:  List = []

            for i in has_step:
                snap = curr[i]
                result = snap['result']
                if result != 'Ongoing':
                    winner = snap.get('winner_id', '')
                    if winner == self_ids[i]:
                        cand_values[i][k_idx] = 1.0
                    elif winner == opp_ids[i]:
                        cand_values[i][k_idx] = -1.0
                    else:
                        cand_values[i][k_idx] = 0.0
                else:
                    leaf_idx.append(i)
                    leaf_sv.append(snap['state_vector'])
                    leaf_av.append(snap['action_feature_vectors'])
                    leaf_la.append(snap['actions'])

            if leaf_sv:
                leaf_outs = self.self_agent.compute_policy_output_batch(
                    leaf_sv, leaf_av, leaf_la
                )
                for i, out in zip(leaf_idx, leaf_outs):
                    cand_values[i][k_idx] = float(out.value)

        # best candidate 선택
        result_actions = []
        for i in range(n):
            cands = candidates[i]
            vals  = cand_values[i]
            if not cands:
                result_actions.append(policy_outs[i].action)
            else:
                best_k = max(range(len(cands)), key=lambda k: vals[k])
                result_actions.append(cands[best_k])
        return result_actions


# ---------------------------------------------------------------------------
# MCTSEvalSession
# ---------------------------------------------------------------------------

class MCTSEvalSession:
    def __init__(self, ports: List[int], n_agents: int,
                 self_agent, opp_agent, mcts_cfg: MCTSConfig,
                 host: str = '127.0.0.1', max_turns: int = 100) -> None:
        self.ports      = ports
        self.n_agents   = n_agents
        self.self_agent = self_agent
        self.opp_agent  = opp_agent
        self.mcts_cfg   = mcts_cfg
        self.host       = host
        self.max_turns  = max_turns

    def run_suite(self, prefix: str, total_matches: int,
                  *, progress_callback=None) -> Dict[str, Any]:
        scenarios    = _scenario_definitions(prefix)
        n_scenarios  = len(scenarios)
        per_scenario = max(1, total_matches // n_scenarios)
        total_games  = per_scenario * n_scenarios

        workers = [
            _MCTSPortWorker(p, self.n_agents, self.self_agent, self.opp_agent,
                            self.mcts_cfg, self.host, self.max_turns)
            for p in self.ports
        ]
        for w in workers: w.start()
        for w in workers: w.wait_ready()

        m          = len(self.ports)
        total_envs = m * self.n_agents
        per_scen:  Dict[int, List] = {i: [] for i in range(n_scenarios)}
        games_done = 0
        it         = 0
        t0         = time.time()

        while games_done < total_games:
            slot_per_worker: List[List] = [[] for _ in range(m)]
            for w_idx in range(m):
                for slot in range(self.n_agents):
                    slot_abs = it * total_envs + w_idx * self.n_agents + slot
                    scen_idx = slot_abs % n_scenarios
                    scen     = scenarios[scen_idx]
                    slot_per_worker[w_idx].append({
                        'scenario_idx':   scen_idx,
                        'label':          scen['label'],
                        'side_name':      scen['side_name'],
                        'self_deck_name': scen['self_deck_name'],
                        'opp_deck_name':  scen['opp_deck_name'],
                        'relation_name':  scen['relation_name'],
                        'p1_deck':        scen['p1_deck'],
                        'p2_deck':        scen['p2_deck'],
                        'p1_id':          scen['p1_id'],
                        'p2_id':          scen['p2_id'],
                        'self_id':        scen['self_id'],
                        'opp_id':         scen['opp_id'],
                    })

            for w_idx, worker in enumerate(workers):
                worker.submit(slot_per_worker[w_idx])

            for worker in workers:
                batch = worker.get_result()
                for r in batch:
                    si = r['scenario_idx']
                    if len(per_scen[si]) < per_scenario:
                        per_scen[si].append(r)
                        games_done += 1
                        if progress_callback:
                            lr = 'Win' if r['self_won'] else ('Draw' if r['draw'] else 'Loss')
                            progress_callback(games_done, total_games, lr, r['scenario_label'])
            it += 1

        for w in workers: w.stop()

        results = []
        for si, scen in enumerate(scenarios):
            gs        = per_scen[si]
            eps       = len(gs)
            self_wins = sum(1 for g in gs if g['self_won'])
            opp_wins  = sum(1 for g in gs if g['opp_won'])
            draws     = sum(1 for g in gs if g['draw'])
            results.append({
                'label':            scen['label'],
                'side_name':        scen['side_name'],
                'self_deck_name':   scen['self_deck_name'],
                'opp_deck_name':    scen['opp_deck_name'],
                'relation_name':    scen['relation_name'],
                'matches':          eps,
                'self_wins':        self_wins,
                'opp_wins':         opp_wins,
                'draws':            draws,
                'win_rate_percent': _human_rate(self_wins, eps),
                'avg_steps':        sum(g['steps']       for g in gs) / max(eps, 1),
                'avg_final_turn':   sum(g['final_turn']  for g in gs) / max(eps, 1),
            })

        agg         = _summarize(results)
        first_rows  = [r for r in results if r['side_name'] == '선공']
        second_rows = [r for r in results if r['side_name'] == '후공']
        same_rows   = [r for r in results if r['relation_name'] == '같은 덱']
        diff_rows   = [r for r in results if r['relation_name'] == '다른 덱']
        or_rows     = [r for r in results if r['self_deck_name'] == '귤']
        ch_rows     = [r for r in results if r['self_deck_name'] == '샤를로테']
        avg = lambda rows: sum(r['win_rate_percent'] for r in rows) / len(rows) if rows else 0.0

        return {
            'label':        prefix,
            'results':      results,
            'aggregate':    agg,
            'first_avg':    avg(first_rows),
            'second_avg':   avg(second_rows),
            'side_gap':     avg(first_rows) - avg(second_rows),
            'same_avg':     avg(same_rows),
            'diff_avg':     avg(diff_rows),
            'orange_avg':   avg(or_rows),
            'charlotte_avg':avg(ch_rows),
            'best':         max(results, key=lambda x: float(x['win_rate_percent']), default=None),
            'worst':        min(results, key=lambda x: float(x['win_rate_percent']), default=None),
            'elapsed':      time.time() - t0,
            'games_done':   games_done,
        }


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def _human_rate(n: int, d: int) -> float:
    return 0.0 if d <= 0 else 100.0 * n / d


def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    eps       = sum(int(r['matches'])    for r in results)
    self_wins = sum(int(r['self_wins'])  for r in results)
    opp_wins  = sum(int(r['opp_wins'])   for r in results)
    draws     = sum(int(r['draws'])      for r in results)
    avg_steps = sum(float(r['avg_steps'])     * int(r['matches']) for r in results) / max(eps, 1)
    avg_turns = sum(float(r['avg_final_turn'])* int(r['matches']) for r in results) / max(eps, 1)

    def _avg_where(key: str, val: str) -> float:
        rows = [r for r in results if str(r.get(key, '')) == val]
        return sum(float(r['win_rate_percent']) for r in rows) / len(rows) if rows else 0.0

    return {
        'episodes':              eps,
        'self_wins':             self_wins,
        'opp_wins':              opp_wins,
        'draws':                 draws,
        'self_win_rate_percent': _human_rate(self_wins, eps),
        'opp_win_rate_percent':  _human_rate(opp_wins,  eps),
        'side_gap_percent':      _human_rate(self_wins - opp_wins, eps),
        'avg_steps':             avg_steps,
        'avg_final_turn':        avg_turns,
        'same_avg':              _avg_where('relation_name', '같은 덱'),
        'diff_avg':              _avg_where('relation_name', '다른 덱'),
        'orange_avg':            _avg_where('self_deck_name', '귤'),
        'charlotte_avg':         _avg_where('self_deck_name', '샤를로테'),
    }


def _format_report(title: str, suite: Dict[str, Any]) -> str:
    agg = suite['aggregate']
    lines = [
        f'=== {title} ===',
        f"label={suite['label']}",
        f"episodes={agg['episodes']}",
        f"self_wins={agg['self_wins']}",
        f"opp_wins={agg['opp_wins']}",
        f"draws={agg['draws']}",
        f"self_win_rate_percent={agg['self_win_rate_percent']:.2f}",
        f"opp_win_rate_percent={agg['opp_win_rate_percent']:.2f}",
        f"side_gap_percent={agg['side_gap_percent']:.2f}",
        f"avg_steps={agg['avg_steps']:.2f}",
        f"avg_final_turn={agg['avg_final_turn']:.2f}",
        f"same_avg={agg['same_avg']:.2f}",
        f"diff_avg={agg['diff_avg']:.2f}",
        f"orange_avg={agg['orange_avg']:.2f}",
        f"charlotte_avg={agg['charlotte_avg']:.2f}",
        f"first_avg={suite['first_avg']:.2f}",
        f"second_avg={suite['second_avg']:.2f}",
        '',
    ]
    for row in suite['results']:
        lines.append(
            f"- {row['label']}: self={row['self_wins']}, opp={row['opp_wins']}, d={row['draws']}, "
            f"wr={float(row['win_rate_percent']):.1f}%, "
            f"avg_steps={float(row['avg_steps']):.1f}, "
            f"avg_turn={float(row['avg_final_turn']):.1f}"
        )
    if suite.get('best') is not None:
        lines += ['', f"best={suite['best']['label']} ({float(suite['best']['win_rate_percent']):.1f}%)"]
    if suite.get('worst') is not None:
        lines.append(f"worst={suite['worst']['label']} ({float(suite['worst']['win_rate_percent']):.1f}%)")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def run_mcts_eval(
    *,
    ports: List[int],
    n_agents: int,
    self_kind: str,
    self_model_path: Optional[str],
    opp_kind: str,
    opp_model_path: Optional[str],
    total_matches: int,
    depth: int,
    top_k: int,
    device: str,
    seed: Optional[int],
    host: str,
    max_turns: int,
    log_file: Optional[str],
) -> None:
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(log_file) if log_file else Path('log') / f'mcts_{ts}.log'
    _setup_logger(log_path)

    _ensure_importable()

    print(f'[*] mcts_session_client started  pid={os.getpid()}')
    print(f'[*] ports={ports}  M={len(ports)}  N={n_agents}  total_matches={total_matches}')
    print(f'[*] device={device}  seed={seed}  depth={depth}  top_k={top_k}')
    print(f'[*] self={self_kind}({self_model_path or "-"})  opp={opp_kind}({opp_model_path or "-"})')
    if depth == 0:
        print('[*] depth=0: MCTS 비활성 (pure policy, eval_client.py와 동등)')

    t_start = time.time()

    if self_kind == 'rl':
        if not self_model_path:
            raise ValueError('--self rl requires --model-path')
        self_agent = _build_rl_agent(self_model_path, device, seed)
    else:
        self_agent = _build_agent(self_kind, device=device, seed=seed)

    opp_agent = _build_agent(opp_kind, model_path=opp_model_path,
                             device=device, seed=(seed or 0) + 1,
                             self_agent=self_agent)

    print(f'[*] self_agent={self_agent.name}  opp_agent={opp_agent.name}')

    mcts_cfg = MCTSConfig(depth=depth, top_k=top_k)
    prefix   = 'mcts' if depth > 0 else 'policy'
    t_prog   = time.time()

    def _progress(done: int, total: int, result: str, matchup: str) -> None:
        interval = max(1, total // 20)
        if done != 1 and done != total and done % interval != 0:
            return
        elapsed = max(1e-9, time.time() - t_prog)
        print(
            f'[*] mcts progress: {done}/{total} eps/s={done / elapsed:.2f} | '
            f'last={result} | matchup={matchup}',
            flush=True,
        )

    session = MCTSEvalSession(
        ports=ports, n_agents=n_agents,
        self_agent=self_agent, opp_agent=opp_agent,
        mcts_cfg=mcts_cfg, host=host, max_turns=max_turns,
    )
    suite   = session.run_suite(prefix, total_matches, progress_callback=_progress)

    title   = f'mcts_session: {self_agent.name}(d={depth},k={top_k}) vs {opp_agent.name}'
    report  = _format_report(title, suite)
    elapsed = suite['elapsed']
    speed   = suite['games_done'] / max(elapsed, 0.001)

    print()
    print(report)
    print()
    print(f'[*] eval speed: {speed:.2f} eps/s  elapsed: {_format_elapsed(elapsed)}')

    rp = Path('log') / f'mcts_{ts}_summary.txt'
    summary = '\n'.join([
        '=== mcts_session_client Summary ===',
        f'ports={",".join(str(p) for p in ports)}',
        f'n_agents={n_agents}',
        f'total_matches={total_matches}',
        f'depth={depth}  top_k={top_k}',
        f'device={device}  seed={seed}',
        f'self={self_agent.name}',
        f'opp={opp_agent.name}',
        '',
        report,
        '',
        f'avg_speed={speed:.2f} eps/s',
        f'elapsed={_format_elapsed(time.time() - t_start)}',
        f'log={log_path}',
    ]) + '\n'
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(summary, encoding='utf-8')
    print(f'[*] Summary: {rp}')
    print(f'[*] Total elapsed: {_format_elapsed(time.time() - t_start)}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='MCTS depth-limited value search 병렬 평가 클라이언트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--multi',          action='store_true')
    p.add_argument('--m-servers',      type=int,  default=4)
    p.add_argument('--port',           type=int,  default=9000)
    p.add_argument('--base-port',      type=int,  default=9000)
    p.add_argument('--host',           type=str,  default='127.0.0.1')
    p.add_argument('--n-agents',       type=int,  default=32)

    p.add_argument('--self',           type=str,  default='rl',  dest='self_kind',
                   choices=['rl', 'random', 'greedy', 'rule_based'])
    p.add_argument('--model-path',     type=str,  default=None)
    p.add_argument('--opp',            type=str,  default='self', dest='opp_kind',
                   choices=['self', 'rl', 'random', 'greedy', 'rule_based'])
    p.add_argument('--opp-model-path', type=str,  default=None)

    p.add_argument('--depth',          type=int,  default=1,
                   help='MCTS lookahead 깊이 (0=MCTS 없음, 1=1수 앞, 2=2수 앞, ...)')
    p.add_argument('--top-k',          type=int,  default=2,
                   help='탐색할 후보 액션 수')
    p.add_argument('--total-matches',  type=int,  default=400)
    p.add_argument('--max-turns',      type=int,  default=100)
    p.add_argument('--device',         type=str,  default='cpu')
    p.add_argument('--seed',           type=int,  default=7)
    p.add_argument('--log-file',       type=str,  default=None)
    return p.parse_args()


def main() -> None:
    args  = _parse_args()
    ports = (
        [args.base_port + i for i in range(args.m_servers)]
        if args.multi else [args.port]
    )

    model_path = args.model_path
    if not model_path and args.self_kind == 'rl':
        candidates = sorted(
            list(Path('models').glob('ckpt_ep*.pt')) + list(Path('models').glob('final_model.pt')),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            model_path = str(candidates[-1])
            print(f'[*] auto-selected model: {model_path}')
        else:
            raise FileNotFoundError('No model .pt found in models/. Use --model-path.')

    run_mcts_eval(
        ports=ports,
        n_agents=args.n_agents,
        self_kind=args.self_kind,
        self_model_path=model_path,
        opp_kind=args.opp_kind,
        opp_model_path=args.opp_model_path,
        total_matches=args.total_matches,
        depth=args.depth,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
        host=args.host,
        max_turns=args.max_turns,
        log_file=args.log_file,
    )


if __name__ == '__main__':
    _t0 = time.perf_counter()
    try:
        main()
    except KeyboardInterrupt:
        print('\n[*] interrupted')
    except Exception as exc:
        import traceback
        print(f'[!] mcts_session_client failed: {exc}')
        print(traceback.format_exc())
        raise SystemExit(1) from exc
    finally:
        print(f'[*] runtime: {_format_elapsed(time.perf_counter() - _t0)}')
