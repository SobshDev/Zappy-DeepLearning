#!/usr/bin/env python3
"""Train Zappy agents — Phase 2: recurrent MAPPO foraging.

Flags are auto-generated from ``TrainConfig`` fields, e.g.::

    python -m zappy_rl.train --run-name forage6 --total-env-steps 10000000
    python -m zappy_rl.train --num-envs 64 --rollout-steps 32 \
        --total-env-steps 100000 --wandb disabled          # tiny smoke run

W&B mode ``auto`` resolves to online when credentials exist, else offline.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

from zappy_rl.algo.mappo import TrainConfig, train


def _wandb_auto() -> str:
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():
        return "online"
    return "offline"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool" or isinstance(f.default, bool):
            ap.add_argument(flag, default=f.default, action=argparse.BooleanOptionalAction)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    ap.add_argument("--run-name", default="forage")
    ap.add_argument("--wandb", default="auto",
                    choices=("auto", "online", "offline", "disabled"))
    args = vars(ap.parse_args())

    run_name = args.pop("run_name")
    wandb_mode = args.pop("wandb")
    if wandb_mode == "auto":
        wandb_mode = _wandb_auto()
    tc = TrainConfig(**args)
    train(tc, run_name=run_name, wandb_mode=wandb_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
