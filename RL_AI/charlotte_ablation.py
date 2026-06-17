"""Charlotte vs Charlotte ablation runner.

This script keeps the current trainer/evaluator stack, but fixes the deck
pair to Charlotte vs Charlotte so we can isolate data-mix effects from model
capacity and reward issues.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

_DOTNET_CANDIDATES = [
    Path(os.environ.get("DOTNET_ROOT", "")),
    Path(os.environ.get("DOTNET_ROOT_X64", "")),
    Path(r"C:\Program Files\dotnet"),
]
for _candidate in _DOTNET_CANDIDATES:
    try:
        if _candidate and _candidate.exists() and (_candidate / "dotnet.exe").exists():
            os.environ.setdefault("DOTNET_ROOT", str(_candidate))
            os.environ.setdefault("DOTNET_ROOT_X64", str(_candidate))
            break
    except Exception:
        pass

import torch

from RL_AI.agents import (
    SeaEngineGreedyAgent,
    SeaEngineRandomAgent,
    SeaEngineRLAgent,
    SeaEngineRuleBasedAgent,
    default_model_hidden_dim,
    infer_hidden_dim_from_state_dict,
    load_state_dict_flexible,
)
from RL_AI.analysis.reports import build_win_rate_report, save_report
from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM
from RL_AI.training.trainer import SeaEnginePPOTrainer
from RL_AI.training.experiment import _build_training_opponent_schedule, _format_plan_counts
from train_client import _PortWorker, _format_elapsed, _raise_worker_errors, _layout_label, _worst_layout_win_rate, run_stage2_refinement
os.chdir(PROJECT_ROOT)

CHARLOTTE_DECK = json.dumps(["Cl_L", "Cl_B", "Cl_N", "Cl_R", "Cl_P", "Cl_P", "Cl_P"])
TCP_CHARLOTTE_LAYOUT_IDS = (3, 7)


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
    f = open(log_file, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f"[*] log file: {log_file}")


def _resolve_project_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def _build_trainer(seed: Optional[int], device: str, model_path: Optional[str] = None) -> SeaEnginePPOTrainer:
    if seed is not None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    agent = SeaEngineRLAgent(device=device, seed=seed)
    if model_path:
        checkpoint = Path(_resolve_project_path(model_path))
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=agent.device)
        hidden_dim = infer_hidden_dim_from_state_dict(state_dict, fallback=agent.hidden_dim)
        agent.hidden_dim = hidden_dim
        agent.ensure_model(STATE_VECTOR_DIM)
        assert agent.model is not None
        load_state_dict_flexible(agent.model, state_dict)
        agent.model.eval()
    return SeaEnginePPOTrainer(agent)


def _build_opponent_pool(names: Sequence[str], seed: Optional[int]) -> List[object]:
    mapping = {
        "random": SeaEngineRandomAgent(seed=seed),
        "greedy": SeaEngineGreedyAgent(seed=None if seed is None else seed + 1),
        "rule_based": SeaEngineRuleBasedAgent(seed=None if seed is None else seed + 2),
    }
    pool: List[object] = []
    for name in names:
        key = str(name).strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"unknown opponent: {name!r}. Use random/greedy/rule_based.")
        pool.append(mapping[key])
    if not pool:
        pool = [mapping["random"], mapping["greedy"], mapping["rule_based"]]
    return pool


def _compact_eval_line(name: str, summary: Dict[str, object]) -> str:
    episodes = int(summary.get("episodes", 0) or 0)
    p1_wins = int(summary.get("p1_wins", 0) or 0)
    p2_wins = int(summary.get("p2_wins", 0) or 0)
    draws = int(summary.get("draws", 0) or 0)
    avg_steps = float(summary.get("avg_steps", 0.0) or 0.0)
    avg_final_turn = float(summary.get("avg_final_turn", 0.0) or 0.0)
    wr = 0.0 if episodes == 0 else p1_wins / episodes
    return (
        f"{name}=episodes={episodes}, p1_wins={p1_wins}, p2_wins={p2_wins}, draws={draws}, "
        f"wr={wr:.3f}, avg_steps={avg_steps:.2f}, avg_final_turn={avg_final_turn:.2f}, "
        f"report={summary.get('report_path', '')}"
    )


def _evaluate_suite(
    trainer: SeaEnginePPOTrainer,
    *,
    card_data_path: str,
    max_turns: int,
    eval_matches: int,
    seed: Optional[int],
    label: str,
) -> Dict[str, Dict[str, object]]:
    opponents = {
        "random": SeaEngineRandomAgent(seed=_seed_with_offset(seed, 101)),
        "greedy": SeaEngineGreedyAgent(seed=_seed_with_offset(seed, 202)),
        "rule_based": SeaEngineRuleBasedAgent(seed=_seed_with_offset(seed, 252)),
    }
    results: Dict[str, Dict[str, object]] = {}
    for opp_name, opp_agent in opponents.items():
        summary = trainer.evaluate(
            opponent_agent=opp_agent,
            num_matches=eval_matches,
            card_data_path=card_data_path,
            player1_deck=CHARLOTTE_DECK,
            player2_deck=CHARLOTTE_DECK,
            max_turns=max_turns,
            include_history=False,
            match_context={
                "mode_label": "CharlotteOnly",
                "side_label": "First",
                "self_deck_label": "Charlotte",
                "opp_deck_label": "Charlotte",
                "relation_label": "fixed",
            },
        )
        results[opp_name] = summary
        print(f"[{label}] {_compact_eval_line(opp_name, summary)}")
    return results


def _seed_with_offset(seed: Optional[int], offset: int) -> Optional[int]:
    return None if seed is None else seed + offset


def _resolve_tcp_ports(*, multi: bool, m_servers: int, base_port: int, port: int) -> List[int]:
    if multi:
        count = max(1, int(m_servers))
        return [int(base_port) + i for i in range(count)]
    return [int(port)]


def _charlotte_layout_id(global_index: int) -> int:
    return TCP_CHARLOTTE_LAYOUT_IDS[global_index % len(TCP_CHARLOTTE_LAYOUT_IDS)]


def _start_tcp_workers(
    *,
    ports: Sequence[int],
    n_agents: int,
    trainer: SeaEnginePPOTrainer,
    host: str,
    max_turns: int,
    reward_mode: str = "dense_if_full",
    mcts_depth: int = 0,
    mcts_top_k: int = 1,
) -> List[_PortWorker]:
    workers = [
        _PortWorker(
            port,
            n_agents,
            trainer,
            host,
            max_turns=max_turns,
            reward_mode=reward_mode,
            mcts_depth=mcts_depth,
            mcts_top_k=mcts_top_k,
        )
        for port in ports
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.wait_ready()
    print(f"[*] TCP workers ready: ports={list(ports)}  n_agents={n_agents}")
    return workers


def _run_tcp_charlotte_eval(
    *,
    trainer: SeaEnginePPOTrainer,
    workers: Sequence[_PortWorker],
    max_turns: int,
    eval_matches: int,
    seed: Optional[int],
    label: str,
    reward_mode: str = "dense_if_full",
) -> Dict[str, Dict[str, object]]:
    from RL_AI.agents import SeaEngineGreedyAgent, SeaEngineRandomAgent, SeaEngineRuleBasedAgent

    if eval_matches <= 0:
        return {}

    eval_opponents = {
        "random": SeaEngineRandomAgent(seed=_seed_with_offset(seed, 101)),
        "greedy": SeaEngineGreedyAgent(seed=_seed_with_offset(seed, 202)),
        "rule_based": SeaEngineRuleBasedAgent(seed=_seed_with_offset(seed, 252)),
    }
    total_envs = sum(worker.n_agents for worker in workers)
    base_offsets: List[int] = []
    running = 0
    for worker in workers:
        base_offsets.append(running)
        running += worker.n_agents

    prev_sample_actions = getattr(trainer.agent, "sample_actions", None)
    if prev_sample_actions is not None:
        trainer.agent.sample_actions = False

    try:
        results: Dict[str, Dict[str, object]] = {}
        for opp_name, opp_agent in eval_opponents.items():
            wins = losses = draws = 0
            layout_stats: Dict[str, Dict[str, object]] = {}
            for layout_id in TCP_CHARLOTTE_LAYOUT_IDS:
                lw = ll = ld = 0
                remaining = max(1, int(eval_matches))
                it = 0
                while remaining > 0:
                    for w_idx, worker in enumerate(workers):
                        worker.submit(
                            it,
                            base_offsets[w_idx],
                            total_envs,
                            opp_agent,
                            layouts=[layout_id] * worker.n_agents,
                        )
                    for worker in workers:
                        batch_results = worker.get_result()
                        _raise_worker_errors(batch_results, f"{label}/{opp_name}/{_layout_label(layout_id)}")
                        take = min(remaining, len(batch_results))
                        for row in batch_results[:take]:
                            if row.get("ai_won"):
                                lw += 1
                            elif str(row.get("result", "")) in {"Draw", "Ongoing"}:
                                ld += 1
                            else:
                                ll += 1
                        remaining -= take
                    it += 1
                lname = _layout_label(layout_id)
                ln = lw + ll + ld
                layout_stats[lname] = {
                    "wins": lw,
                    "losses": ll,
                    "draws": ld,
                    "episodes": ln,
                    "win_rate": lw / max(1, ln),
                }
                wins += lw
                losses += ll
                draws += ld
            n_games = wins + losses + draws
            wr = wins / max(1, n_games)
            results[opp_name] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "episodes": n_games,
                "win_rate": wr,
                "layout_stats": layout_stats,
                "games_per_layout": max(1, int(eval_matches)),
            }
            print(f"[{label}]  n={n_games:4d}  vs {opp_name}: w/l/d={wins}/{losses}/{draws}  wr={wr:.3f}")
            for lname, row in layout_stats.items():
                print(
                    f"[{label}]      {opp_name} {lname}: "
                    f"w/l/d={row['wins']}/{row['losses']}/{row['draws']}  "
                    f"wr={row['win_rate']:.3f}"
                )
        return results
    finally:
        if prev_sample_actions is not None:
            trainer.agent.sample_actions = prev_sample_actions


def _run_tcp_charlotte_ablation(
    *,
    train_episodes: int,
    eval_matches: int,
    checkpoint_eval_matches: int,
    n_agents: int,
    ports: Sequence[int],
    host: str,
    update_interval: int,
    save_interval: int,
    checkpoint_interval: int,
    early_stage2: bool,
    early_stage2_patience: int,
    early_stage2_min_episode: int,
    early_stage2_min_delta: float,
    early_stage2_worst_win_rate: float,
    stage2_episodes: int,
    stage2_num_envs: int,
    stage2_mcts_depth: int,
    stage2_mcts_top_k: int,
    stage2_output_path: str,
    stage2_card_data_path: str,
    log_interval: int,
    max_turns: int,
    seed: Optional[int],
    device: str,
    card_data_path: str,
    train_opponents: Sequence[str],
    resume_model_path: Optional[str],
    output_model_path: str,
    reward_mode: str = "dense_if_full",
) -> Dict[str, object]:
    log_path = Path("log") / f"charlotte_ablation_tcp_{Path(output_model_path).stem}_{seed or 'noseed'}.log"
    _setup_logger(log_path)

    trainer = _build_trainer(seed, device, model_path=resume_model_path)
    opponent_pool = _build_opponent_pool(train_opponents, seed)

    print("[*] Charlotte-only ablation")
    print("[*] backend=tcp")
    print(f"[*] deck=Charlotte vs Charlotte")
    print(f"[*] train_episodes={train_episodes}  eval_matches={eval_matches}")
    print(f"[*] checkpoint_eval_matches={checkpoint_eval_matches}")
    print(f"[*] ports={list(ports)}  n_agents={n_agents}")
    print(f"[*] update_interval={update_interval}  save_interval={save_interval}  checkpoint_interval={checkpoint_interval}")
    print(f"[*] reward_mode={reward_mode}")
    print(
        f"[*] early_stage2={early_stage2}  patience={early_stage2_patience} "
        f"min_episode={early_stage2_min_episode} min_delta={early_stage2_min_delta}"
    )
    print(
        f"[*] stage2_episodes={stage2_episodes}  stage2_num_envs={stage2_num_envs} "
        f"stage2_mcts_depth={stage2_mcts_depth} stage2_mcts_top_k={stage2_mcts_top_k}"
    )
    print(f"[*] stage2_output_path={_resolve_project_path(stage2_output_path)}")
    print(f"[*] stage2_card_data_path={_resolve_project_path(stage2_card_data_path)}")
    print(f"[*] max_turns={max_turns}  device={device}  seed={seed}")
    print(f"[*] train_opponents={[getattr(a, 'name', str(a)) for a in opponent_pool]}")
    print(f"[*] card_data_path={_resolve_project_path(card_data_path)}")

    workers = _start_tcp_workers(
        ports=ports,
        n_agents=n_agents,
        trainer=trainer,
        host=host,
        max_turns=max_turns,
        reward_mode=reward_mode,
        mcts_depth=0,
        mcts_top_k=1,
    )

    base_offsets: List[int] = []
    running = 0
    for worker in workers:
        base_offsets.append(running)
        running += worker.n_agents
    total_envs = sum(worker.n_agents for worker in workers)

    pre_eval = _run_tcp_charlotte_eval(
        trainer=trainer,
        workers=workers,
        max_turns=max_turns,
        eval_matches=eval_matches,
        seed=seed,
        label="pre-eval",
        reward_mode=reward_mode,
    )

    schedule, plan_counts = _build_training_opponent_schedule(
        opponent_pool=opponent_pool,
        train_episodes=train_episodes,
        num_envs=total_envs,
        save_interval=max(1, int(save_interval)),
        seed=seed,
    )
    print(f"[train] Opponent plan: {_format_plan_counts(plan_counts)}")

    from RL_AI.training.storage import RolloutBuffer

    episodes_done = 0
    wins = losses = draws = 0
    updates = 0
    last_losses: Dict[str, float] = {}
    opponent_stats: Counter[str] = Counter()
    layout_stats: Counter[str] = Counter()
    result_stats: Counter[str] = Counter()
    steps_total = 0
    final_turn_total = 0
    checkpoint_reports: List[Dict[str, object]] = []
    next_checkpoint = max(1, int(checkpoint_interval))
    best_checkpoint_score: Optional[float] = None
    stage2_triggered = False
    stage2_handoff_checkpoint = ""
    t0 = time.time()

    try:
        iterations = (train_episodes + total_envs - 1) // total_envs
        for it in range(iterations):
            if episodes_done >= train_episodes:
                break

            collect_started_at = time.perf_counter()
            batch_opp_names: List[str] = []
            for w_idx, worker in enumerate(workers):
                ep_start = it * total_envs + base_offsets[w_idx]
                slot_opps = []
                slot_layouts = []
                for j in range(worker.n_agents):
                    idx = ep_start + j
                    opp_name = schedule[idx] if idx < len(schedule) else "random"
                    batch_opp_names.append(opp_name)
                    slot_opps.append(next((a for a in opponent_pool if a.name == opp_name), opponent_pool[0]))
                    slot_layouts.append(_charlotte_layout_id(idx))
                worker.submit(it, base_offsets[w_idx], total_envs, slot_opps, layouts=slot_layouts)

            batch_results: List[Dict[str, object]] = []
            for worker in workers:
                batch_results.extend(worker.get_result())
            _raise_worker_errors(batch_results, "Charlotte TCP collect")
            collect_elapsed = max(1e-9, time.perf_counter() - collect_started_at)

            merged = RolloutBuffer()
            for row in batch_results:
                for step in row["buffer"].steps:
                    merged.add_step(step)

            update_started_at = time.perf_counter()
            last_losses = trainer.update_from_buffer(merged)
            update_elapsed = max(1e-9, time.perf_counter() - update_started_at)
            updates += 1

            batch_episodes = len(batch_results)
            episodes_done += batch_episodes
            batch_wins = 0
            batch_losses = 0
            batch_draws = 0
            batch_steps = 0
            batch_turns = 0
            batch_opponents = Counter()
            batch_layouts = Counter()
            batch_results_by_name = Counter()

            for row in batch_results:
                if row.get("ai_won"):
                    batch_wins += 1
                elif str(row.get("result", "")) in {"Draw", "Ongoing"}:
                    batch_draws += 1
                else:
                    batch_losses += 1
                batch_steps += int(row.get("steps", 0) or 0)
                batch_turns += int(row.get("final_turn", 0) or 0)
                batch_results_by_name[str(row.get("result", ""))] += 1
                layout_id = int(row.get("layout_id", -1))
                batch_layouts[_layout_label(layout_id)] += 1
            for opp_name in batch_opp_names:
                batch_opponents[str(opp_name)] += 1

            wins += batch_wins
            losses += batch_losses
            draws += batch_draws
            steps_total += batch_steps
            final_turn_total += batch_turns
            opponent_stats.update(batch_opponents)
            layout_stats.update(batch_layouts)
            result_stats.update(batch_results_by_name)

            if log_interval > 0 and (
                episodes_done % log_interval < total_envs or episodes_done >= train_episodes
            ):
                elapsed = max(1e-9, time.time() - t0)
                wr = wins / max(1, episodes_done)
                print(
                    f"[train] ep={episodes_done:6d}/{train_episodes} "
                    f"wr={wr:.3f} "
                    f"loss_p={last_losses.get('policy_loss', 0.0):+.4f} "
                    f"loss_v={last_losses.get('value_loss', 0.0):.4f} "
                    f"kl={last_losses.get('approx_kl', 0.0):+.4f} "
                    f"clip={last_losses.get('clip_fraction', 0.0):.3f} "
                    f"grad_norm={last_losses.get('grad_norm', 0.0):.4f} "
                    f"ep/s={episodes_done / elapsed:.2f} "
                    f"collect={collect_elapsed:.1f}s  update={update_elapsed:.2f}s  "
                    f"avg_steps={steps_total / max(1, episodes_done):.1f}  "
                    f"avg_turn={final_turn_total / max(1, episodes_done):.1f}  "
                    f"draw={draws}  timeout={result_stats.get('Ongoing', 0)}  "
                    f"elapsed={_format_elapsed(elapsed)}"
                )

            if checkpoint_interval > 0:
                while episodes_done >= next_checkpoint:
                    ckpt_path = Path(_resolve_project_path(f"models/charlotte_ablation_ckpt_ep{episodes_done}.pt"))
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    assert trainer.agent.model is not None
                    torch.save(trainer.agent.model.state_dict(), ckpt_path)
                    print(f"[*] Checkpoint: {ckpt_path}")

                    print(f"[*] Checkpoint evaluation ... ep={episodes_done}")
                    ckpt_eval = _run_tcp_charlotte_eval(
                        trainer=trainer,
                        workers=workers,
                        max_turns=max_turns,
                        eval_matches=checkpoint_eval_matches,
                        seed=seed,
                        label=f"ckpt-{episodes_done}",
                        reward_mode=reward_mode,
                    )
                    ckpt_score = 0.0
                    if ckpt_eval:
                        ckpt_score = sum(row["win_rate"] for row in ckpt_eval.values()) / max(1, len(ckpt_eval))
                    ckpt_worst = _worst_layout_win_rate(ckpt_eval)
                    if early_stage2:
                        if ckpt_worst >= early_stage2_worst_win_rate:
                            stage2_triggered = True
                            stage2_handoff_checkpoint = str(ckpt_path)
                            print(
                                f"[*] Early-Stage2 worst-layout stop: ep={episodes_done} "
                                f"worst={ckpt_worst:.4f} threshold={early_stage2_worst_win_rate:.4f}"
                            )
                            print(
                                f"[*] Early-Stage2 triggered at ep={episodes_done}. "
                                f"checkpoint={ckpt_path}"
                            )
                        elif best_checkpoint_score is None:
                            best_checkpoint_score = ckpt_score
                            print(
                                f"[*] Early-Stage2 monitor init: ep={episodes_done} "
                                f"score={ckpt_score:.4f}"
                            )
                        elif ckpt_score >= best_checkpoint_score + early_stage2_min_delta:
                            print(
                                f"[*] Early-Stage2 monitor improve: ep={episodes_done} "
                                f"score={ckpt_score:.4f} best={best_checkpoint_score:.4f}"
                            )
                            best_checkpoint_score = ckpt_score
                        elif ckpt_score < best_checkpoint_score:
                            stage2_triggered = True
                            stage2_handoff_checkpoint = str(ckpt_path)
                            print(
                                f"[*] Early-Stage2 monitor drop: ep={episodes_done} "
                                f"score={ckpt_score:.4f} best={best_checkpoint_score:.4f}"
                            )
                            print(
                                f"[*] Early-Stage2 triggered at ep={episodes_done}. "
                                f"checkpoint={ckpt_path}"
                            )
                        else:
                            print(
                                f"[*] Early-Stage2 monitor hold: ep={episodes_done} "
                                f"score={ckpt_score:.4f} best={best_checkpoint_score:.4f}"
                            )
                    checkpoint_reports.append({
                        "episodes": episodes_done,
                        "checkpoint_path": str(ckpt_path),
                        "score": ckpt_score,
                        "worst_score": ckpt_worst,
                        "eval": ckpt_eval,
                    })
                    print(f"[*] Checkpoint score: {ckpt_score:.4f}")
                    next_checkpoint += max(1, int(checkpoint_interval))
                    if stage2_triggered:
                        break
                if stage2_triggered:
                    break
            if stage2_triggered:
                break
    finally:
        for worker in workers:
            try:
                worker.stop()
            except Exception:
                pass

    stage2_result: Optional[Dict[str, object]] = None
    if stage2_triggered and stage2_handoff_checkpoint:
        print("[*] Stage2 handoff ...")
        run_stage2_refinement(
            checkpoint_path=stage2_handoff_checkpoint,
            output_path=stage2_output_path,
            episodes=stage2_episodes,
            num_envs=stage2_num_envs,
            seed=seed,
            device=device,
            card_data_path=stage2_card_data_path,
            max_turns=max_turns,
            log_interval=log_interval,
            ports=ports,
            host=host,
            mcts_depth=stage2_mcts_depth,
            mcts_top_k=stage2_mcts_top_k,
        )
        stage2_result = {
            "checkpoint_path": stage2_handoff_checkpoint,
            "output_path": _resolve_project_path(stage2_output_path),
            "episodes": stage2_episodes,
        }
        out_path = Path(_resolve_project_path(stage2_output_path))
        post_eval: Dict[str, Dict[str, object]] = {}
    else:
        out_path = Path(_resolve_project_path(output_model_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assert trainer.agent.model is not None
        torch.save(trainer.agent.model.state_dict(), out_path)
        print(f"[*] Saved final model: {out_path}")

        print("[*] Post-training evaluation ...")
        eval_workers = _start_tcp_workers(
            ports=ports,
            n_agents=n_agents,
            trainer=trainer,
            host=host,
            max_turns=max_turns,
            reward_mode=reward_mode,
            mcts_depth=0,
            mcts_top_k=1,
        )
        try:
            post_eval = _run_tcp_charlotte_eval(
                trainer=trainer,
                workers=eval_workers,
                max_turns=max_turns,
                eval_matches=eval_matches,
                seed=seed,
                label="post-eval",
                reward_mode=reward_mode,
            )
        finally:
            for worker in eval_workers:
                try:
                    worker.stop()
                except Exception:
                    pass

    summary_path = Path("log") / "charlotte_ablation_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Charlotte Only Ablation (TCP) ===\n")
        f.write("deck=Charlotte vs Charlotte\n")
        f.write("backend=tcp\n")
        f.write(f"train_episodes={train_episodes}\n")
        f.write(f"eval_matches={eval_matches}\n")
        f.write(f"checkpoint_eval_matches={checkpoint_eval_matches}\n")
        f.write(f"n_agents={n_agents}\n")
        f.write(f"ports={list(ports)}\n")
        f.write(f"update_interval={update_interval}\n")
        f.write(f"save_interval={save_interval}\n")
        f.write(f"checkpoint_interval={checkpoint_interval}\n")
        f.write(f"early_stage2={early_stage2}\n")
        f.write(f"early_stage2_patience={early_stage2_patience}\n")
        f.write(f"early_stage2_min_episode={early_stage2_min_episode}\n")
        f.write(f"early_stage2_min_delta={early_stage2_min_delta}\n")
        f.write(f"stage2_episodes={stage2_episodes}\n")
        f.write(f"stage2_num_envs={stage2_num_envs}\n")
        f.write(f"stage2_mcts_depth={stage2_mcts_depth}\n")
        f.write(f"stage2_mcts_top_k={stage2_mcts_top_k}\n")
        f.write(f"stage2_output_path={stage2_output_path}\n")
        f.write(f"stage2_card_data_path={stage2_card_data_path}\n")
        f.write(f"max_turns={max_turns}\n")
        f.write(f"seed={seed}\n")
        f.write(f"device={device}\n")
        f.write(f"train_opponents={[getattr(a, 'name', str(a)) for a in opponent_pool]}\n")
        f.write("\n")
        for key in ("random", "greedy", "rule_based"):
            if key in pre_eval:
                f.write(_compact_eval_line(f"before/{key}", pre_eval[key]) + "\n")
        f.write("\n")
        f.write(
            "train="
            f"episodes={episodes_done} | "
            f"wins={wins} | "
            f"losses={losses} | "
            f"draws={draws} | "
            f"updates={updates} | "
            f"avg_steps={steps_total / max(1, episodes_done):.2f} | "
            f"avg_turn={final_turn_total / max(1, episodes_done):.2f}\n"
        )
        f.write(f"opponent_stats={dict(opponent_stats)}\n")
        f.write(f"layout_stats={dict(layout_stats)}\n")
        f.write(f"checkpoint_reports={checkpoint_reports}\n")
        f.write(f"stage2_triggered={stage2_triggered}\n")
        f.write(f"stage2_handoff_checkpoint={stage2_handoff_checkpoint}\n")
        f.write(f"stage2_result={stage2_result}\n")
        f.write("\n")
        for key in ("random", "greedy", "rule_based"):
            if key in post_eval:
                f.write(_compact_eval_line(f"after/{key}", post_eval[key]) + "\n")

    report_path = save_report(
        "\n".join(
            [
                "Charlotte Only Ablation (TCP)",
                "",
                *(_compact_eval_line(f"before/{k}", pre_eval[k]) for k in ("random", "greedy", "rule_based") if k in pre_eval),
                "",
                f"train={{'episodes': {episodes_done}, 'wins': {wins}, 'losses': {losses}, 'draws': {draws}, 'updates': {updates}}}",
                "",
                *(_compact_eval_line(f"after/{k}", post_eval[k]) for k in ("random", "greedy", "rule_based") if k in post_eval),
                "",
                f"model={out_path}",
            ]
        ),
        Path("log") / f"charlotte_ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    print(f"[*] Summary: {summary_path}")
    print(f"[*] Report: {report_path}")
    return {
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "train": {
            "episodes": episodes_done,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "updates": updates,
        },
        "checkpoint_reports": checkpoint_reports,
        "stage2_triggered": stage2_triggered,
        "stage2_handoff_checkpoint": stage2_handoff_checkpoint,
        "stage2_result": stage2_result,
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "model_path": str(out_path),
    }


def run_charlotte_ablation(
    *,
    train_episodes: int,
    eval_matches: int,
    checkpoint_eval_matches: int,
    num_envs: int,
    n_agents: Optional[int] = None,
    update_interval: int,
    save_interval: int,
    checkpoint_interval: int,
    early_stage2: bool,
    early_stage2_patience: int,
    early_stage2_min_episode: int,
    early_stage2_min_delta: float,
    stage2_episodes: int,
    stage2_num_envs: int,
    stage2_mcts_depth: int,
    stage2_mcts_top_k: int,
    stage2_output_path: str,
    stage2_card_data_path: str,
    log_interval: int,
    max_turns: int,
    seed: Optional[int],
    device: str,
    card_data_path: str,
    train_opponents: Sequence[str],
    resume_model_path: Optional[str],
    output_model_path: str,
    reward_mode: str = "dense_if_full",
    backend: str = "tcp",
    multi: bool = True,
    m_servers: int = 2,
    base_port: int = 9000,
    port: int = 9000,
    host: str = "127.0.0.1",
) -> Dict[str, object]:
    if str(backend).strip().lower() == "tcp":
        ports = _resolve_tcp_ports(multi=multi, m_servers=m_servers, base_port=base_port, port=port)
        return _run_tcp_charlotte_ablation(
            train_episodes=train_episodes,
            eval_matches=eval_matches,
            checkpoint_eval_matches=checkpoint_eval_matches,
            n_agents=int(n_agents if n_agents is not None else num_envs),
            ports=ports,
            host=host,
            update_interval=update_interval,
            save_interval=save_interval,
            checkpoint_interval=checkpoint_interval,
            early_stage2=early_stage2,
            early_stage2_patience=early_stage2_patience,
            early_stage2_min_episode=early_stage2_min_episode,
            early_stage2_min_delta=early_stage2_min_delta,
            early_stage2_worst_win_rate=early_stage2_worst_win_rate,
            stage2_episodes=stage2_episodes,
            stage2_num_envs=stage2_num_envs,
            stage2_mcts_depth=stage2_mcts_depth,
            stage2_mcts_top_k=stage2_mcts_top_k,
            stage2_output_path=stage2_output_path,
            stage2_card_data_path=stage2_card_data_path,
            log_interval=log_interval,
            max_turns=max_turns,
            seed=seed,
            device=device,
            card_data_path=card_data_path,
            train_opponents=train_opponents,
            resume_model_path=resume_model_path,
            output_model_path=output_model_path,
            reward_mode=reward_mode,
        )

    log_path = Path("log") / f"charlotte_ablation_{Path(output_model_path).stem}_{seed or 'noseed'}.log"
    _setup_logger(log_path)

    trainer = _build_trainer(seed, device, model_path=resume_model_path)
    opponent_pool = _build_opponent_pool(train_opponents, seed)

    print("[*] Charlotte-only ablation")
    print(f"[*] deck=Charlotte vs Charlotte")
    print(f"[*] train_episodes={train_episodes}  eval_matches={eval_matches}")
    print(f"[*] checkpoint_eval_matches={checkpoint_eval_matches}")
    print(f"[*] num_envs={num_envs}  update_interval={update_interval}  save_interval={save_interval}")
    print(f"[*] reward_mode={reward_mode}")
    print(f"[*] max_turns={max_turns}  device={device}  seed={seed}")
    print(f"[*] train_opponents={[getattr(a, 'name', str(a)) for a in opponent_pool]}")
    print(f"[*] card_data_path={_resolve_project_path(card_data_path)}")

    pre_eval = _evaluate_suite(
        trainer,
        card_data_path=card_data_path,
        max_turns=max_turns,
        eval_matches=eval_matches,
        seed=seed,
        label="pre-eval",
    )

    train_started = time.perf_counter()
    train_result = trainer.train(
        num_episodes=train_episodes,
        opponent_pool=list(opponent_pool),
        card_data_path=card_data_path,
        player1_deck=CHARLOTTE_DECK,
        player2_deck=CHARLOTTE_DECK,
        max_turns=max_turns,
        update_interval=update_interval,
        save_interval=save_interval,
        log_interval=log_interval,
        num_envs=num_envs,
    )
    train_elapsed = max(0.0, time.perf_counter() - train_started)

    out_path = Path(_resolve_project_path(output_model_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert trainer.agent.model is not None
    torch.save(trainer.agent.model.state_dict(), out_path)
    print(f"[*] Saved final model: {out_path}")

    post_eval = _evaluate_suite(
        trainer,
        card_data_path=card_data_path,
        max_turns=max_turns,
        eval_matches=eval_matches,
        seed=seed,
        label="post-eval",
    )

    summary_path = Path("log") / "charlotte_ablation_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Charlotte Only Ablation ===\n")
        f.write(f"deck=Charlotte vs Charlotte\n")
        f.write(f"train_episodes={train_episodes}\n")
        f.write(f"eval_matches={eval_matches}\n")
        f.write(f"checkpoint_eval_matches={checkpoint_eval_matches}\n")
        f.write(f"num_envs={num_envs}\n")
        f.write(f"update_interval={update_interval}\n")
        f.write(f"save_interval={save_interval}\n")
        f.write(f"max_turns={max_turns}\n")
        f.write(f"seed={seed}\n")
        f.write(f"device={device}\n")
        f.write(f"train_opponents={[getattr(a, 'name', str(a)) for a in opponent_pool]}\n")
        f.write("\n")
        f.write(_compact_eval_line("before/random", pre_eval["random"]) + "\n")
        f.write(_compact_eval_line("before/greedy", pre_eval["greedy"]) + "\n")
        f.write(_compact_eval_line("before/rule_based", pre_eval["rule_based"]) + "\n")
        f.write("\n")
        f.write(
            "train="
            f"episodes={train_result.get('episodes', 0)} | "
            f"wins={train_result.get('wins', 0)} | "
            f"losses={train_result.get('losses', 0)} | "
            f"draws={train_result.get('draws', 0)} | "
            f"updates={train_result.get('updates', 0)} | "
            f"avg_steps={float(train_result.get('avg_steps', 0.0)):.2f} | "
            f"avg_turn={float(train_result.get('avg_final_turn', 0.0)):.2f} | "
            f"elapsed={train_elapsed:.1f}s\n"
        )
        f.write(f"opponent_stats={train_result.get('opponent_stats', {})}\n")
        f.write(f"layout_stats={train_result.get('layout_stats', {})}\n")
        f.write("\n")
        f.write(_compact_eval_line("after/random", post_eval["random"]) + "\n")
        f.write(_compact_eval_line("after/greedy", post_eval["greedy"]) + "\n")
        f.write(_compact_eval_line("after/rule_based", post_eval["rule_based"]) + "\n")

    report_path = save_report(
        "\n".join(
            [
                "Charlotte Only Ablation",
                "",
                _compact_eval_line("before/random", pre_eval["random"]),
                _compact_eval_line("before/greedy", pre_eval["greedy"]),
                _compact_eval_line("before/rule_based", pre_eval["rule_based"]),
                "",
                f"train={train_result}",
                "",
                _compact_eval_line("after/random", post_eval["random"]),
                _compact_eval_line("after/greedy", post_eval["greedy"]),
                _compact_eval_line("after/rule_based", post_eval["rule_based"]),
                "",
                f"model={out_path}",
            ]
        ),
        Path("log") / f"charlotte_ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    print(f"[*] Summary: {summary_path}")
    print(f"[*] Report: {report_path}")
    return {
        "pre_eval": pre_eval,
        "post_eval": post_eval,
        "train": train_result,
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "model_path": str(out_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Charlotte vs Charlotte ablation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-episodes", type=int, default=1024)
    parser.add_argument("--eval-matches", type=int, default=50)
    parser.add_argument("--checkpoint-eval-matches", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--n-agents", type=int, default=64, help="TCP mode only: agents per server.")
    parser.add_argument("--port", type=int, default=9000, help="Single-server TCP mode port.")
    parser.add_argument("--m-servers", type=int, default=2, help="TCP multi-server count.")
    parser.add_argument("--base-port", type=int, default=9000, help="TCP multi-server base port.")
    parser.add_argument("--multi", action="store_true", help="Use TCP multi-server mode.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TCP server host.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("tcp", "pythonnet"),
        default="tcp",
        help="TCP backend is fastest; pythonnet keeps the legacy local path.",
    )
    parser.add_argument("--update-interval", type=int, default=16)
    parser.add_argument("--save-interval", type=int, default=0, help="0 disables self-play checkpoints during ablation.")
    parser.add_argument("--checkpoint-interval", type=int, default=512, help="Checkpoint/eval interval in episodes.")
    parser.add_argument(
        "--early-stage2",
        dest="early_stage2",
        action="store_true",
        help="Enable checkpoint plateau handoff to Stage 2.",
    )
    parser.add_argument(
        "--no-early-stage2",
        dest="early_stage2",
        action="store_false",
        help="Disable checkpoint plateau handoff to Stage 2.",
    )
    parser.set_defaults(early_stage2=True)
    parser.add_argument("--early-stage2-patience", type=int, default=2)
    parser.add_argument("--early-stage2-min-episode", type=int, default=512)
    parser.add_argument("--early-stage2-min-delta", type=float, default=0.01)
    parser.add_argument("--early-stage2-worst-win-rate", type=float, default=0.80)
    parser.add_argument("--stage2-episodes", type=int, default=10000)
    parser.add_argument("--stage2-num-envs", type=int, default=32)
    parser.add_argument("--stage2-mcts-depth", type=int, default=1)
    parser.add_argument("--stage2-mcts-top-k", type=int, default=2)
    parser.add_argument("--stage2-output-path", type=str, default="models/charlotte_ablation_stage2.pt")
    parser.add_argument("--stage2-card-data-path", type=str, default="cards/Cards.csv")
    parser.add_argument("--log-interval", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=70)
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--card-data-path", type=str, default="cards/Cards.csv")
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="dense_if_full",
        choices=("terminal", "terminal_action", "dense_if_full"),
        help="TCP reward mode used for Charlotte-only ablation.",
    )
    parser.add_argument("--resume-model-path", type=str, default="")
    parser.add_argument("--output-model-path", type=str, default="models/charlotte_ablation_final.pt")
    parser.add_argument(
        "--train-opponents",
        nargs="+",
        default=["random", "greedy", "rule_based"],
        help="Training opponent pool names.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "eval", "both"),
        default="both",
    )
    parser.add_argument(
        "--eval-model-path",
        type=str,
        default="models/charlotte_ablation_final.pt",
        help="Model path used when mode=eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "eval":
        if str(args.backend).strip().lower() == "tcp":
            ports = _resolve_tcp_ports(multi=args.multi, m_servers=args.m_servers, base_port=args.base_port, port=args.port)
            _setup_logger(Path("log") / f"charlotte_ablation_eval_{Path(args.eval_model_path).stem}_{args.seed or 'noseed'}.log")
            trainer = _build_trainer(args.seed, args.device, model_path=args.eval_model_path)
            workers = _start_tcp_workers(
                ports=ports,
                n_agents=args.n_agents,
                trainer=trainer,
                host=args.host,
                max_turns=args.max_turns,
                reward_mode=args.reward_mode,
                mcts_depth=0,
                mcts_top_k=1,
            )
            print("[*] Charlotte-only evaluation")
            try:
                _run_tcp_charlotte_eval(
                    trainer=trainer,
                    workers=workers,
                    max_turns=args.max_turns,
                    eval_matches=args.eval_matches,
                    seed=args.seed,
                    label="eval",
                    reward_mode=args.reward_mode,
                )
            finally:
                for worker in workers:
                    try:
                        worker.stop()
                    except Exception:
                        pass
        else:
            trainer = _build_trainer(args.seed, args.device, model_path=args.eval_model_path)
            _setup_logger(Path("log") / f"charlotte_ablation_eval_{Path(args.eval_model_path).stem}_{args.seed or 'noseed'}.log")
            print("[*] Charlotte-only evaluation")
            _evaluate_suite(
                trainer,
                card_data_path=args.card_data_path,
                max_turns=args.max_turns,
                eval_matches=args.eval_matches,
                seed=args.seed,
                label="eval",
            )
        return

    run_charlotte_ablation(
        train_episodes=args.train_episodes,
        eval_matches=args.eval_matches,
        checkpoint_eval_matches=args.checkpoint_eval_matches,
        num_envs=args.num_envs,
        n_agents=args.n_agents,
        update_interval=args.update_interval,
        save_interval=args.save_interval,
        checkpoint_interval=args.checkpoint_interval,
        early_stage2=args.early_stage2,
        early_stage2_patience=args.early_stage2_patience,
        early_stage2_min_episode=args.early_stage2_min_episode,
        early_stage2_min_delta=args.early_stage2_min_delta,
        early_stage2_worst_win_rate=args.early_stage2_worst_win_rate,
        stage2_episodes=args.stage2_episodes,
        stage2_num_envs=args.stage2_num_envs,
        stage2_mcts_depth=args.stage2_mcts_depth,
        stage2_mcts_top_k=args.stage2_mcts_top_k,
        stage2_output_path=args.stage2_output_path,
        stage2_card_data_path=args.stage2_card_data_path,
        log_interval=args.log_interval,
        max_turns=args.max_turns,
        seed=args.seed,
        device=args.device,
        card_data_path=args.card_data_path,
        reward_mode=args.reward_mode,
        train_opponents=args.train_opponents,
        resume_model_path=args.resume_model_path or None,
        output_model_path=args.output_model_path,
        backend=args.backend,
        multi=args.multi,
        m_servers=args.m_servers,
        base_port=args.base_port,
        port=args.port,
        host=args.host,
    )


if __name__ == "__main__":
    main()
