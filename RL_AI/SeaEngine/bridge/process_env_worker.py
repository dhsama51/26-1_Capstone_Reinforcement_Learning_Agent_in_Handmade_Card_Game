"""Worker process entrypoint for isolated SeaEngine vector environments."""

from __future__ import annotations

import contextlib
import os
import random
import traceback
from typing import Any, Dict, Optional


_INTERNAL_OPPONENTS = {"random", "greedy", "rule_based"}


def _suppress_worker_output() -> None:
    quiet_worker = os.getenv("SEAENGINE_QUIET_WORKER_LOG", "1") == "1"
    if not quiet_worker:
        return
    devnull = open(os.devnull, "w", encoding="utf-8")
    with contextlib.suppress(Exception):
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)


def _card_id(card: Dict[str, Any] | None) -> str:
    if not card:
        return ""
    return str(card.get("card_id", card.get("id", card.get("name", ""))))


def _role(card: Dict[str, Any] | None) -> str:
    if not card:
        return ""
    role = str(card.get("role", "")).strip()
    if role:
        return role
    suffix = _card_id(card).split("_")[-1].strip()[-1:]
    return {"L": "Leader", "B": "Bishop", "N": "Knight", "R": "Rook", "P": "Pawn"}.get(suffix, "")


def _players(snapshot: Dict[str, Any]) -> tuple[str, str]:
    active = str(snapshot.get("active_player", ""))
    ids = [str(player.get("id", "")) for player in snapshot.get("players", []) if str(player.get("id", ""))]
    enemy = next((pid for pid in ids if pid != active), "")
    return active, enemy


def _leader(snapshot: Dict[str, Any], owner: str) -> Optional[Dict[str, Any]]:
    for card in snapshot.get("board", []):
        if str(card.get("owner", "")) != owner:
            continue
        if bool(card.get("is_placed", False)) and _role(card) == "Leader":
            return card
    return None


def _distance_xy(x1: int, y1: int, x2: int, y2: int) -> int:
    if min(x1, y1, x2, y2) < 0:
        return 99
    return abs(x1 - x2) + abs(y1 - y2)


def _source_card(snapshot: Dict[str, Any], action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_uid = str(action.get("source", ""))
    if not source_uid:
        return None
    for card in snapshot.get("board", []):
        if str(card.get("uid", "")) == source_uid:
            return card
    for player in snapshot.get("players", []):
        for card in player.get("hand", []):
            if str(card.get("uid", "")) == source_uid:
                return card
    return None


def _target_card(snapshot: Dict[str, Any], action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target = action.get("target", {}) or {}
    target_uid = str(target.get("guid", ""))
    if not target_uid:
        return None
    for card in snapshot.get("board", []):
        if str(card.get("uid", "")) == target_uid:
            return card
    return None


def _cell_after_action(action: Dict[str, Any], source: Optional[Dict[str, Any]]) -> tuple[int, int]:
    target = action.get("target", {}) or {}
    if str(target.get("type", "")) == "Cell":
        return int(target.get("pos_x", -1)), int(target.get("pos_y", -1))
    if source:
        return int(source.get("pos_x", -1)), int(source.get("pos_y", -1))
    return -1, -1


def _attackers_on(snapshot: Dict[str, Any], target: Optional[Dict[str, Any]], *, by_owner: str) -> int:
    if not target or not target.get("is_placed"):
        return 0
    tx = int(target.get("pos_x", -1))
    ty = int(target.get("pos_y", -1))
    count = 0
    for card in snapshot.get("board", []):
        if str(card.get("owner", "")) != by_owner or not bool(card.get("is_placed", False)):
            continue
        cx = int(card.get("pos_x", -1))
        cy = int(card.get("pos_y", -1))
        if _distance_xy(cx, cy, tx, ty) <= 2:
            count += 1
    return count


def _safe_actions(snapshot: Dict[str, Any]) -> list[Dict[str, Any]]:
    return list(snapshot.get("actions", []) or [])


def _score_random_action(_snapshot: Dict[str, Any], _action: Dict[str, Any]) -> float:
    return 0.0


def _score_greedy_action(snapshot: Dict[str, Any], action: Dict[str, Any]) -> float:
    effect_id = str(action.get("effect_id", ""))
    target = action.get("target", {}) or {}
    target_type = str(target.get("type", "None"))
    target_card = _target_card(snapshot, action)
    source = _source_card(snapshot, action)
    active, enemy = _players(snapshot)
    enemy_leader = _leader(snapshot, enemy)

    score = 0.0
    if effect_id == "DefaultAttack":
        score += 120.0
    elif effect_id == "DeployUnit":
        score += 70.0
    elif effect_id == "DefaultMove":
        score += 25.0
    elif effect_id == "TurnEnd":
        score -= 150.0
    else:
        score += 55.0

    if target_type == "Unit" and target_card is not None:
        target_hp = float(target_card.get("hp", 0.0))
        target_atk = float(target_card.get("effective_atk", target_card.get("atk", 0.0)))
        source_atk = float((source or {}).get("effective_atk", (source or {}).get("atk", 0.0)))
        if _role(target_card) == "Leader":
            score += 200.0
        if source_atk >= target_hp > 0:
            score += 60.0 + target_atk * 2.0
        score += max(0.0, 18.0 - target_hp)

    if target_type == "Cell":
        tx = int(target.get("pos_x", -1))
        ty = int(target.get("pos_y", -1))
        if enemy_leader is not None:
            score += max(
                0.0,
                16.0 - float(
                    _distance_xy(tx, ty, int(enemy_leader.get("pos_x", -1)), int(enemy_leader.get("pos_y", -1)))
                )
                * 2.0,
            )
        if 2 <= tx <= 3 and 2 <= ty <= 3:
            score += 8.0

    if source is not None:
        source_role = _role(source)
        source_id = _card_id(source)
        if effect_id == "DeployUnit":
            score += {"Leader": 14.0, "Knight": 11.0, "Bishop": 10.0, "Rook": 10.0, "Pawn": 7.0}.get(source_role, 0.0)
        if source_id in {"Or_N", "Cl_B", "Cl_N"}:
            score += 6.0

    own_board = [c for c in snapshot.get("board", []) if str(c.get("owner", "")) == active and bool(c.get("is_placed", False))]
    enemy_board = [c for c in snapshot.get("board", []) if str(c.get("owner", "")) == enemy and bool(c.get("is_placed", False))]
    score += min(len(own_board), 7) * 1.0
    score -= min(len(enemy_board), 7) * 0.5
    return score


def _score_rule_action(snapshot: Dict[str, Any], action: Dict[str, Any]) -> float:
    active, enemy = _players(snapshot)
    own_leader = _leader(snapshot, active)
    enemy_leader = _leader(snapshot, enemy)
    source = _source_card(snapshot, action)
    target_card = _target_card(snapshot, action)
    effect_id = str(action.get("effect_id", ""))
    target = action.get("target", {}) or {}
    target_type = str(target.get("type", "None"))
    source_role = _role(source)
    source_id = _card_id(source)
    source_atk = float((source or {}).get("effective_atk", (source or {}).get("atk", 0.0)))

    score = 0.0
    if effect_id == "TurnEnd":
        non_end_actions = [a for a in snapshot.get("actions", []) if str(a.get("effect_id", "")) != "TurnEnd"]
        return -200.0 - float(len(non_end_actions))
    if effect_id == "DefaultAttack":
        score += 110.0
    elif effect_id == "DeployUnit":
        score += 58.0
    elif effect_id == "DefaultMove":
        score += 28.0
    else:
        score += 72.0

    if target_card is not None:
        target_role = _role(target_card)
        target_hp = float(target_card.get("hp", 0.0))
        target_atk = float(target_card.get("effective_atk", target_card.get("atk", 0.0)))
        if target_role == "Leader":
            score += 190.0
            score += max(0.0, 40.0 - target_hp * 4.0)
        if source_atk >= target_hp > 0:
            score += 55.0 + target_atk * 3.0
            if target_role in {"Bishop", "Knight", "Rook"}:
                score += 12.0
        if source is not None and target_atk >= float(source.get("hp", 0.0)) > 0:
            score -= 18.0
        score += max(0.0, 16.0 - target_hp)
        score += _attackers_on(snapshot, target_card, by_owner=active) * 3.0

    tx, ty = _cell_after_action(action, source)
    if target_type == "Cell" and tx >= 0 and ty >= 0:
        if 2 <= tx <= 3 and 2 <= ty <= 3:
            score += 12.0
        if enemy_leader is not None:
            before = _distance_xy(
                int((source or {}).get("pos_x", -1)),
                int((source or {}).get("pos_y", -1)),
                int(enemy_leader.get("pos_x", -1)),
                int(enemy_leader.get("pos_y", -1)),
            )
            after = _distance_xy(tx, ty, int(enemy_leader.get("pos_x", -1)), int(enemy_leader.get("pos_y", -1)))
            score += max(-10.0, float(before - after) * 8.0)
            if after <= 2:
                score += 18.0
        if own_leader is not None:
            own_after = _distance_xy(tx, ty, int(own_leader.get("pos_x", -1)), int(own_leader.get("pos_y", -1)))
            if source_role != "Leader" and own_after <= 2:
                score += 4.0

    if effect_id == "DeployUnit":
        score += {"Leader": 14.0, "Knight": 13.0, "Bishop": 11.0, "Rook": 10.0, "Pawn": 7.0}.get(source_role, 0.0)
        if source_id in {"Cl_B", "Or_N"}:
            score += 6.0
    if source_id == "Or_N" and target_card is not None:
        score += 8.0
    if source_id in {"Cl_B", "Cl_N", "Cl_R"} and effect_id not in {"TurnEnd", "DefaultMove"}:
        score += 5.0

    own_board = [c for c in snapshot.get("board", []) if str(c.get("owner", "")) == active and bool(c.get("is_placed", False))]
    enemy_board = [c for c in snapshot.get("board", []) if str(c.get("owner", "")) == enemy and bool(c.get("is_placed", False))]
    score += min(len(own_board), 7) * 1.2
    score -= min(len(enemy_board), 7) * 0.6
    return score


def _select_internal_action(
    opponent_name: str,
    snapshot: Dict[str, Any],
    legal_actions: list[Dict[str, Any]],
    rng: random.Random,
) -> tuple[int, Dict[str, Any]]:
    if not legal_actions:
        raise ValueError("No legal actions available.")
    name = str(opponent_name or "").strip().lower()
    if name == "random":
        idx = rng.randrange(len(legal_actions))
        return idx, legal_actions[idx]

    if name == "greedy":
        scored = [(_score_greedy_action(snapshot, action), idx, action) for idx, action in enumerate(legal_actions)]
    else:
        scored = [(_score_rule_action(snapshot, action), idx, action) for idx, action in enumerate(legal_actions)]

    best_score = max(score for score, _, _ in scored)
    best = [(idx, action) for score, idx, action in scored if score == best_score]
    return best[rng.randrange(len(best))]


def _prepare_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Attach observation features in the worker and optionally trim payload."""
    snap = dict(snapshot)

    encode_obs = os.getenv("SEAENGINE_WORKER_ENCODE_OBS", "1").strip().lower() not in {"0", "false", "no", "off"}
    if encode_obs and snap.get("result") == "Ongoing":
        try:
            from RL_AI.SeaEngine.observation import build_observation

            active_player = str(snap.get("active_player", ""))
            obs = build_observation({**snap, "actions": _safe_actions(snap)}, active_player)
            snap["state_vector"] = obs.state_vector
            snap["action_feature_vectors"] = obs.action_feature_vectors
        except Exception:
            # Encoding is an optimization. If it fails, the main process can
            # fall back to its existing observation builder.
            pass

    compact = os.getenv("SEAENGINE_WORKER_COMPACT_SNAPSHOT", "1").strip().lower() not in {"0", "false", "no", "off"}
    if compact:
        keep_keys = {
            "result",
            "winner_id",
            "turn",
            "active_player",
            "board",
            "players",
            "actions",
            "state_vector",
            "action_feature_vectors",
            "_worker_auto_steps",
        }
        snap = {key: value for key, value in snap.items() if key in keep_keys}

    return snap


def _advance_until_ai_turn_or_done(
    *,
    session,
    snapshot: Dict[str, Any],
    ai_player_id: str,
    opponent_name: str,
    opponent_rng: random.Random,
    auto_opponent: bool,
    max_auto_steps: int,
) -> Dict[str, Any]:
    """Let the worker play opponent turns until the next AI turn or game end."""
    auto_steps = 0
    while (
        auto_opponent
        and snapshot.get("result") == "Ongoing"
        and str(snapshot.get("active_player", "")) != str(ai_player_id)
        and auto_steps < max_auto_steps
    ):
        legal_actions = _safe_actions(snapshot)
        if not legal_actions:
            break

        _, action = _select_internal_action(opponent_name, snapshot, legal_actions, opponent_rng)
        snapshot = session.apply_action(str(action.get("uid", "")))
        auto_steps += 1

    prepared = _prepare_snapshot(snapshot)
    prepared["_worker_auto_steps"] = auto_steps
    return prepared


def worker_loop(conn, card_data_path: str | None) -> None:
    _suppress_worker_output()

    from RL_AI.SeaEngine.bridge.pythonnet_session import PythonNetSession

    session = PythonNetSession(card_data_path=card_data_path)

    ai_player_id = "AI"
    opponent_name = ""
    opponent_seed = 0
    auto_opponent = False
    opponent_rng = random.Random(0)
    max_auto_steps = max(1, int(os.getenv("SEAENGINE_WORKER_MAX_AUTO_STEPS", "256") or "256"))

    try:
        session.start()
        while True:
            msg = conn.recv()
            cmd = str(msg.get("cmd", "")).strip().lower()

            if cmd == "ping":
                conn.send({"ok": True, "pong": True})
                continue

            if cmd == "configure":
                ai_player_id = str(msg.get("ai_player_id", ai_player_id))
                opponent_name = str(msg.get("opponent_name", opponent_name))
                opponent_seed = int(msg.get("opponent_seed", opponent_seed) or 0)
                requested_auto = bool(msg.get("auto_opponent", False))
                auto_opponent = requested_auto and opponent_name in _INTERNAL_OPPONENTS
                opponent_rng = random.Random(opponent_seed)
                snapshot = session.snapshot()
                prepared = _advance_until_ai_turn_or_done(
                    session=session,
                    snapshot=snapshot,
                    ai_player_id=ai_player_id,
                    opponent_name=opponent_name,
                    opponent_rng=opponent_rng,
                    auto_opponent=auto_opponent,
                    max_auto_steps=max_auto_steps,
                )
                conn.send({"ok": True, "configured": True, "snapshot": prepared})
                continue

            if cmd == "reset":
                config = dict(msg.get("config", {}) or {})

                ai_player_id = str(config.pop("_ai_player_id", ai_player_id))
                opponent_name = str(config.pop("_opponent_name", opponent_name))
                opponent_seed = int(config.pop("_opponent_seed", opponent_seed) or 0)
                requested_auto = bool(config.pop("_auto_opponent", False))
                auto_opponent = requested_auto and opponent_name in _INTERNAL_OPPONENTS
                opponent_rng = random.Random(opponent_seed)

                snapshot = session.init_game(**config)
                prepared = _advance_until_ai_turn_or_done(
                    session=session,
                    snapshot=snapshot,
                    ai_player_id=ai_player_id,
                    opponent_name=opponent_name,
                    opponent_rng=opponent_rng,
                    auto_opponent=auto_opponent,
                    max_auto_steps=max_auto_steps,
                )
                conn.send({"ok": True, "snapshot": prepared})
                continue

            if cmd == "step":
                action_uid = str(msg.get("action_uid", ""))
                snapshot = session.apply_action(action_uid)
                prepared = _advance_until_ai_turn_or_done(
                    session=session,
                    snapshot=snapshot,
                    ai_player_id=ai_player_id,
                    opponent_name=opponent_name,
                    opponent_rng=opponent_rng,
                    auto_opponent=auto_opponent,
                    max_auto_steps=max_auto_steps,
                )
                conn.send({"ok": True, "snapshot": prepared})
                continue

            if cmd == "close":
                conn.send({"ok": True})
                break

            conn.send({"ok": False, "error": f"unknown cmd: {cmd}"})
    except BaseException:
        error_msg = traceback.format_exc()
        try:
            conn.send({"ok": False, "error": error_msg})
        except Exception:
            pass
        raise
    finally:
        try:
            session.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
