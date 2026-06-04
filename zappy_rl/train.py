#!/usr/bin/env python3
"""Train Zappy agents — recurrent MAPPO (Phase 2 foraging, Phase 3 rituals).

Flags are auto-generated from ``TrainConfig`` fields, e.g.::

    python -m zappy_rl.train --run-name forage6 --total-env-steps 10000000
    python -m zappy_rl.train --num-envs 64 --rollout-steps 32 \
        --total-env-steps 100000 --wandb disabled          # tiny smoke run
    python -m zappy_rl.train --run-name ritual8x8 --width 8 --height 8 \
        --n-agents 2 --max-episode-ticks 8192 --eval-max-ticks 8192 \
        --ent-coef-token 0.01 --total-env-steps 100000000  # Phase 3 ritual

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
    ap.add_argument("--init-actor", default=None, metavar="PARAMS.msgpack",
                    help="warm-start the actor from a prior run's checkpoint")
    args = vars(ap.parse_args())

    run_name = args.pop("run_name")
    wandb_mode = args.pop("wandb")
    init_actor = args.pop("init_actor")
    if wandb_mode == "auto":
        wandb_mode = _wandb_auto()
    tc = TrainConfig(**args)
    train(tc, run_name=run_name, wandb_mode=wandb_mode, init_actor=init_actor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
