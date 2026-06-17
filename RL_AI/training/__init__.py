from __future__ import annotations

"""Lazy exports for RL_AI.training.

This package intentionally avoids importing heavy modules at import time.
The TCP train_client.py path only needs trainer/storage/reward helpers, and
eager imports would pull in SeaEngineSession -> grpc even when the caller does
not use the gRPC backend.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "PPOConfig",
    "PastSelfAgent",
    "SeaEnginePPOTrainer",
    "RolloutBuffer",
    "RolloutStep",
    "dense_reward_from_transition",
    "terminal_reward_for_player",
    "evaluate_agents",
    "play_evaluation_match",
    "run_checkpoint_training_experiment",
    "run_saved_model_balance_experiment",
    "run_train_eval_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {"PPOConfig", "PastSelfAgent", "SeaEnginePPOTrainer"}:
        module = import_module("RL_AI.training.trainer")
        return getattr(module, name)
    if name in {"RolloutBuffer", "RolloutStep"}:
        module = import_module("RL_AI.training.storage")
        return getattr(module, name)
    if name in {"dense_reward_from_transition", "terminal_reward_for_player"}:
        module = import_module("RL_AI.training.reward")
        return getattr(module, name)
    if name in {"evaluate_agents", "play_evaluation_match"}:
        module = import_module("RL_AI.training.evaluator")
        return getattr(module, name)
    if name in {
        "run_checkpoint_training_experiment",
        "run_saved_model_balance_experiment",
        "run_train_eval_experiment",
    }:
        module = import_module("RL_AI.training.experiment")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

