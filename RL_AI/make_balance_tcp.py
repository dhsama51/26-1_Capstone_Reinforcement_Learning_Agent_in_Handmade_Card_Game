#!/usr/bin/env python3
"""TCP version of the 8-combo balance evaluation.

This runner reuses ``train_client.py``'s TCP ``_PortWorker`` and evaluates
one saved RL model against another across the 8 deck/side combinations.

It is intentionally separate from ``make_balance.py`` so we can avoid the
PythonNet / in-process C# bridge and use the same TCP path as training.
"""

from __future__ import annotations

import argparse
import copy
import io
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _ensure_importable() -> None:
    project_parent = str(Path(__file__).resolve().parent.parent)
    if project_parent not in sys.path:
        sys.path.insert(0, project_parent)


def _sanitize_tag(tag: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag.strip())
    return cleaned.strip("_") or "run"


def _setup_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_file, "w", encoding="utf-8", buffering=1)

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

    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    print(f"[*] log file: {log_file}")


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d} ({seconds:.1f}s)"


def _resolve_ports(*, multi: bool, m_servers: int, base_port: int, port: int) -> list[int]:
    if multi or int(m_servers) > 1:
        count = max(1, int(m_servers))
        return [int(base_port) + i for i in range(count)]
    return [int(port)]


def _zip_directory(src: Path, dst_zip: Path) -> None:
    if not src.exists():
        return
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    if dst_zip.exists():
        dst_zip.unlink()
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in src.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(src.parent)))


def _load_rl_agent(model_path: Path, *, device: str = "auto", seed: Optional[int] = None):
    import torch
    from RL_AI.agents import SeaEngineRLAgent, infer_hidden_dim_from_state_dict, load_state_dict_flexible
    from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM

    def _extract_zip_model(zip_path: Path) -> Path:
        cache_root = Path.home() / ".rl_ai_model_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        signature = f"{zip_path.resolve()}|{zip_path.stat().st_size}|{zip_path.stat().st_mtime_ns}"
        extract_dir = cache_root / zip_path.stem / signature.replace(":", "_").replace("|", "_").replace("\\", "_").replace("/", "_")
        marker = extract_dir / ".extracted"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            marker.write_text(signature, encoding="utf-8")
        pt_candidates = sorted(extract_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not pt_candidates:
            raise FileNotFoundError(f"No .pt model found inside zip: {zip_path}")
        preferred = [p for p in pt_candidates if p.name == "best_model.pt"]
        if preferred:
            return preferred[-1]
        preferred = [p for p in pt_candidates if p.name == "model_ep_10000.pt"]
        if preferred:
            return preferred[-1]
        return pt_candidates[-1]

    resolved = Path(model_path)
    if resolved.suffix.lower() == ".zip":
        resolved = _extract_zip_model(resolved)

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    state_dict = torch.load(resolved, map_location=resolved_device, weights_only=True)
    agent = SeaEngineRLAgent(
        hidden_dim=infer_hidden_dim_from_state_dict(state_dict),
        device=resolved_device,
        sample_actions=False,
        seed=seed,
    )
    agent.ensure_model(state_dim=STATE_VECTOR_DIM)
    assert agent.model is not None
    load_state_dict_flexible(agent.model, state_dict)
    agent.model.eval()
    return agent


def _clone_agent(agent):
    return copy.deepcopy(agent)


def _start_workers(*, ports: Sequence[int], n_agents: int, trainer, host: str, max_turns: int):
    _ensure_importable()
    import RL_AI.train_client as tc

    workers = [tc._PortWorker(port, n_agents, trainer, host, max_turns=max_turns, reward_mode="terminal") for port in ports]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.wait_ready()
    return workers


def _evaluate_8_combos(
    *,
    trainer,
    workers,
    opponent_agent,
    total_matches: int,
    max_turns: int,
    seed: Optional[int],
    run_tag: str,
) -> Dict[str, Any]:
    _ensure_importable()
    import RL_AI.train_client as tc

    total_envs = sum(worker.n_agents for worker in workers)
    base_offsets: list[int] = []
    running = 0
    for worker in workers:
        base_offsets.append(running)
        running += worker.n_agents

    scenario_count = len(tc._ALL_LAYOUTS)
    per = max(0, int(total_matches)) // scenario_count
    rem = max(0, int(total_matches)) % scenario_count

    scenario_results: list[Dict[str, Any]] = []
    aggregate = {
        "episodes": 0,
        "ai_wins": 0,
        "opp_wins": 0,
        "draws": 0,
        "avg_steps_weighted_sum": 0.0,
        "avg_final_turn_weighted_sum": 0.0,
    }

    print(f"[*] TCP balance: run_tag={run_tag}")
    print(f"[*] TCP balance: ports={[w.port for w in workers]}  total_envs={total_envs}  combos={scenario_count}  per_combo={per}  rem={rem}")

    # Opponent is frozen during evaluation, so one clone per worker is enough.
    worker_opponents = [_clone_agent(opponent_agent) for _ in workers]

    layout_ids = list(range(scenario_count))
    start_t = time.perf_counter()
    for idx, layout_id in enumerate(layout_ids):
        scenario_matches = per + (1 if idx < rem else 0)
        if scenario_matches <= 0:
            continue

        layout_label = tc._layout_label(layout_id)
        wins = losses = draws = 0
        total_steps = 0
        total_turns = 0
        remaining = scenario_matches
        it = 0
        while remaining > 0:
            for w_idx, worker in enumerate(workers):
                worker.submit(
                    it,
                    base_offsets[w_idx],
                    total_envs,
                    worker_opponents[w_idx],
                    layouts=[layout_id] * worker.n_agents,
                )
            for worker in workers:
                batch = worker.get_result()
                _raise_worker_errors(batch, f"TCP balance/{layout_label}")
                take = min(remaining, len(batch))
                for row in batch[:take]:
                    if row.get("ai_won"):
                        wins += 1
                    elif str(row.get("result", "")) in {"Draw", "Ongoing"}:
                        draws += 1
                    else:
                        losses += 1
                    total_steps += int(row.get("steps", 0))
                    total_turns += int(row.get("final_turn", 0))
                remaining -= take
            it += 1

        episodes = wins + losses + draws
        wr = wins / max(1, episodes)
        avg_steps = total_steps / max(1, episodes)
        avg_turn = total_turns / max(1, episodes)
        aggregate["episodes"] += episodes
        aggregate["ai_wins"] += wins
        aggregate["opp_wins"] += losses
        aggregate["draws"] += draws
        aggregate["avg_steps_weighted_sum"] += avg_steps * episodes
        aggregate["avg_final_turn_weighted_sum"] += avg_turn * episodes
        scenario_results.append(
            {
                "layout_id": layout_id,
                "label": layout_label,
                "matches": episodes,
                "ai_wins": wins,
                "opp_wins": losses,
                "draws": draws,
                "win_rate": wr,
                "avg_steps": avg_steps,
                "avg_final_turn": avg_turn,
            }
        )
        print(
            f"[balance-tcp] {layout_label}: w/l/d={wins}/{losses}/{draws} "
            f"wr={wr:.3f} avg_steps={avg_steps:.1f} avg_turn={avg_turn:.1f}"
        )

    total = max(1, aggregate["episodes"])
    aggregate["avg_steps"] = aggregate["avg_steps_weighted_sum"] / total
    aggregate["avg_final_turn"] = aggregate["avg_final_turn_weighted_sum"] / total
    elapsed = max(1e-9, time.perf_counter() - start_t)
    aggregate["elapsed_sec"] = elapsed
    aggregate["episodes_per_sec"] = aggregate["episodes"] / elapsed
    aggregate["win_rate"] = aggregate["ai_wins"] / total
    aggregate["loss_rate"] = aggregate["opp_wins"] / total
    aggregate["draw_rate"] = aggregate["draws"] / total

    summary_lines = [
        f"=== TCP Balance ({run_tag}) ===",
        f"model={getattr(trainer.agent, 'name', 'rl')}",
        f"opponent={getattr(opponent_agent, 'name', 'opp')}",
        f"ports={[w.port for w in workers]}",
        f"total_matches={total_matches}",
        f"episodes={aggregate['episodes']}",
        f"win_rate={aggregate['win_rate']:.4f}",
        f"loss_rate={aggregate['loss_rate']:.4f}",
        f"draw_rate={aggregate['draw_rate']:.4f}",
        f"avg_steps={aggregate['avg_steps']:.2f}",
        f"avg_final_turn={aggregate['avg_final_turn']:.2f}",
        f"eps_per_sec={aggregate['episodes_per_sec']:.2f}",
        "",
    ]
    for row in scenario_results:
        summary_lines.append(
            f"{row['label']}: w/l/d={row['ai_wins']}/{row['opp_wins']}/{row['draws']} "
            f"wr={row['win_rate']:.4f} avg_steps={row['avg_steps']:.1f} avg_turn={row['avg_final_turn']:.1f}"
        )

    return {
        "aggregate": aggregate,
        "scenario_results": scenario_results,
        "summary_text": "\n".join(summary_lines) + "\n",
    }


def _raise_worker_errors(results: Sequence[Dict[str, Any]], context: str) -> None:
    errors = [r for r in results if r.get("__worker_error__")]
    if not errors:
        return
    details = []
    for err in errors:
        details.append(
            f"port={err.get('port')} error={err.get('error')}\n{err.get('traceback', '')}"
        )
    raise RuntimeError(f"{context} worker failed:\n" + "\n".join(details))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCP 8-combo balance runner for two saved RL models")
    p.add_argument("--multi", action="store_true", help="Use multi-server TCP mode")
    p.add_argument("--m-servers", type=int, default=2, help="Number of RL_Server processes")
    p.add_argument("--port", type=int, default=9000, help="Single-server port")
    p.add_argument("--base-port", type=int, default=9000, help="First TCP port in multi mode")
    p.add_argument("--host", type=str, default="127.0.0.1", help="RL_Server host")
    p.add_argument("--n-agents", type=int, default=32, help="Concurrent agents per TCP server")
    p.add_argument("--total-matches", type=int, default=4000, help="Total matches across 8 combos (4000 = 500 each)")
    p.add_argument("--max-turns", type=int, default=70, help="Maximum turns per game")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--device", type=str, default="auto", help="torch device")
    p.add_argument("--model-path", type=str, default="", help="Left model checkpoint path")
    p.add_argument("--opponent-model-path", type=str, default="", help="Right model checkpoint path")
    p.add_argument("--run-tag", type=str, default="", help="Output tag for logs/zip")
    p.add_argument("--log-file", type=str, default="", help="Optional log file path")
    return p.parse_args()


def main() -> int:
    _ensure_importable()
    import RL_AI.train_client as tc
    from RL_AI.agents import SeaEngineRLAgent
    from RL_AI.analysis.reports import save_report
    from RL_AI.training.trainer import SeaEnginePPOTrainer

    args = _parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = Path(args.model_path).stem if args.model_path else "model"
    opp_tag = Path(args.opponent_model_path).stem if args.opponent_model_path else model_tag
    run_tag = _sanitize_tag(args.run_tag or f"{model_tag}_vs_{opp_tag}")

    log_dir = Path("log") / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else log_dir / f"make_balance_tcp_{ts}.log"
    _setup_logger(log_file)

    tc._set_layout_filter("both", "both")

    ports = _resolve_ports(multi=args.multi, m_servers=args.m_servers, base_port=args.base_port, port=args.port)
    print("[*] TCP balance launched")
    print(f"[*] run_tag={run_tag}")
    print(f"[*] ports={ports}  M={len(ports)}  N={args.n_agents}")
    print(f"[*] total_matches={args.total_matches}  max_turns={args.max_turns}")
    print(f"[*] model_path={args.model_path or '(required)'}")
    print(f"[*] opponent_model_path={args.opponent_model_path or '(self)'}")

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(model_path or "(missing --model-path)")
    opponent_path = Path(args.opponent_model_path) if args.opponent_model_path else model_path
    if not opponent_path.exists():
        raise FileNotFoundError(opponent_path)

    p1_agent = _load_rl_agent(model_path, device=args.device, seed=args.seed)
    p2_agent = _load_rl_agent(opponent_path, device=args.device, seed=None if args.seed is None else args.seed + 2001)
    trainer = SeaEnginePPOTrainer(p1_agent)

    workers = _start_workers(ports=ports, n_agents=args.n_agents, trainer=trainer, host=args.host, max_turns=args.max_turns)
    try:
        result = _evaluate_8_combos(
            trainer=trainer,
            workers=workers,
            opponent_agent=p2_agent,
            total_matches=args.total_matches,
            max_turns=args.max_turns,
            seed=args.seed,
            run_tag=run_tag,
        )
    finally:
        for worker in workers:
            worker.stop()

    summary_path = log_dir / "make_balance_tcp_summary.txt"
    save_report(result["summary_text"], summary_path)
    zip_path = Path("log") / f"make_balance_tcp_{run_tag}.zip"
    _zip_directory(log_dir, zip_path)

    print(result["summary_text"], end="")
    print(f"[*] summary: {summary_path}")
    print(f"[*] archived logs: {zip_path}")
    print("[*] make_balance_tcp.py finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
