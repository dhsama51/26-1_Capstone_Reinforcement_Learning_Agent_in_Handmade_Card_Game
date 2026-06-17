from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional


def _ensure_importable() -> None:
    from pathlib import Path as _Path
    import sys

    pkg_parent = str(_Path(__file__).resolve().parent.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Game.RL_Server checkpoint evaluation client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--multi", action="store_true", help="Multi-server mode")
    p.add_argument("--m-servers", type=int, default=2, help="Number of RL_Server processes")
    p.add_argument("--port", type=int, default=9000, help="Single-server port")
    p.add_argument("--base-port", type=int, default=9000, help="First port in multi mode")
    p.add_argument("--host", type=str, default="127.0.0.1", help="RL_Server host")
    p.add_argument("--n-agents", type=int, default=16, help="Concurrent agents per server")
    p.add_argument("--n-eval", type=int, default=64, help="Evaluation games per opponent")
    p.add_argument("--max-turns", type=int, default=70, help="Maximum turns per game")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--device", type=str, default="auto", help="torch device")
    p.add_argument(
        "--model-path",
        type=str,
        default="models/final_model.pt",
        help="Checkpoint path to evaluate",
    )
    p.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    return p.parse_args()


def _load_agent(model_path: Path, device: str, seed: Optional[int]):
    import torch
    from RL_AI.agents import SeaEngineRLAgent, infer_hidden_dim_from_state_dict, load_state_dict_flexible
    from RL_AI.SeaEngine.observation import STATE_VECTOR_DIM

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    hidden_dim = infer_hidden_dim_from_state_dict(state_dict)
    agent = SeaEngineRLAgent(hidden_dim=hidden_dim, device=device, sample_actions=False, seed=seed)
    agent.ensure_model(STATE_VECTOR_DIM)
    load_state_dict_flexible(agent.model, state_dict)
    assert agent.model is not None
    agent.model.eval()
    return agent


def main() -> None:
    _ensure_importable()

    import os
    import time
    from RL_AI import train_client as tc
    from RL_AI.training.trainer import SeaEnginePPOTrainer

    args = _parse_args()
    ports = [args.base_port + i for i in range(args.m_servers)] if args.multi else [args.port]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else Path("log") / f"eval_{ts}.log"
    tc._setup_logger(log_path)

    print(f"[*] eval_client started  pid={os.getpid()}")
    print(f"[*] ports={ports}  M={len(ports)}  N={args.n_agents}  n_eval={args.n_eval}  device={args.device}")
    print(f"[*] model_path={args.model_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    agent = _load_agent(model_path, args.device, args.seed)
    trainer = SeaEnginePPOTrainer(agent)

    session = tc.TrainingSession(
        ports=ports,
        n_agents=args.n_agents,
        trainer=trainer,
        device=args.device,
        host=args.host,
        log_interval=0,
        save_interval=0,
        total_episodes=0,
        seed=args.seed,
        eval_interval=0,
        n_eval=args.n_eval,
        max_turns=args.max_turns,
        reward_mode="terminal_action",
    )

    workers = [
        tc._PortWorker(port, args.n_agents, trainer, args.host, args.max_turns)
        for port in ports
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.wait_ready()

    try:
        t0 = time.time()
        stats = session._run_eval(workers, 0, args.n_eval, label="eval")
        elapsed = time.time() - t0
        print(f"[*] eval finished in {elapsed:.2f}s")
        for opp_name, row in stats.items():
            print(
                f"[eval] {opp_name}: w/l/d={row['wins']}/{row['losses']}/{row['draws']} "
                f"wr={row['win_rate']:.3f}"
            )
    finally:
        for worker in workers:
            worker.stop()


if __name__ == "__main__":
    main()
