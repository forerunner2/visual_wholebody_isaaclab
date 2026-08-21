# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the low-level B1+Z1 policy with the original custom RSL-RL stack (v1).

This script drives the ported custom ActorCritic / PPO / OnPolicyRunner from
``visual_wholebody_isaaclab.learning.rsl_rl``, preserving the original project's
algorithm design: prop/priv/hist observation split, dual control heads,
priv_encoder + StateHistoryEncoder, priv_reg distillation and the internal DAgger
(hist_encoding) update loop.

    python scripts/rsl_rl/train_v1.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --num_envs 1024 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from datetime import datetime

import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the low-level policy with the original custom rsl_rl stack.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0",
    help="Name of the task (defaults to the low-level B1Z1 policy task).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
parser.add_argument("--log_dir", type=str, default=None, help="Overwrite the log directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from visual_wholebody_isaaclab.learning.rsl_rl import (
    OnPolicyRunner,
    VisualWholeBodyVecEnvWrapper,
    get_b1z1_ppo_cfg,
)

import visual_wholebody_isaaclab.tasks  # noqa: F401


def main():
    # load the env config from the registry and override the number of environments
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    # build the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # wrap into the v1-style VecEnv interface expected by the custom runner
    env = VisualWholeBodyVecEnvWrapper(env.unwrapped)

    # load the PPO config (original B1Z1RoughCfgPPO values)
    train_cfg = get_b1z1_ppo_cfg()
    if args_cli.max_iterations is not None:
        train_cfg["runner"]["max_iterations"] = args_cli.max_iterations
    if args_cli.seed is not None:
        train_cfg["seed"] = args_cli.seed

    # specify the log directory
    log_dir = args_cli.log_dir or os.path.join(
        "logs",
        "rsl_rl",
        train_cfg["runner"]["experiment_name"],
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {os.path.abspath(log_dir)}")

    # create runner
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device="cuda:0")
    if args_cli.resume:
        print(f"[INFO] Resuming from: {args_cli.resume}")
        runner.load(args_cli.resume)

    # run training
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
