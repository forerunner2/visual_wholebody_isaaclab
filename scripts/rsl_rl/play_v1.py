# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a low-level B1+Z1 checkpoint trained with the custom RSL-RL stack (v1).

This is the *v1* counterpart of ``scripts/rsl_rl/play.py``. The low-level policy
is trained with ``scripts/rsl_rl/train_v1.py`` using the ported custom
``ActorCritic``/``PPO`` (prop/priv/hist split, dual heads, ``priv_encoder`` +
``StateHistoryEncoder``), whose checkpoint format differs from the stock rsl_rl
v3 workflow — so the stock ``play.py`` cannot load it.

Usage:

    python scripts/rsl_rl/play_v1.py \
        --checkpoint logs/rsl_rl/b1z1_low_level_v1/<run>/model_<iter>.pt \
        --num_envs 1 \
        [--real-time] [--headless]
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play the low-level B1+Z1 v1 policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0",
    help="Name of the task (defaults to the low-level B1Z1 policy task).",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a v1 checkpoint (model_<iter>.pt).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--print-interval", type=int, default=100, help="Print rollout stats every N steps.")
# append AppLauncher cli args (--headless, --device, --video, ...)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from visual_wholebody_isaaclab.learning.low_level_policy import LowLevelPolicy

import visual_wholebody_isaaclab.tasks  # noqa: F401


def main():
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required: pass the path to a v1 checkpoint (model_<iter>.pt).")

    # load the env config from the registry and override the number of environments
    spec = gym.spec(args_cli.task)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    # keep randomization off for a clean rollout (optional; comment out to keep the training defaults)
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False

    # build the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped

    # load the frozen v1 policy (deployment mode: hist_encoding=True, no privileged info)
    policy = LowLevelPolicy(
        checkpoint_path=args_cli.checkpoint,
        obs_space=744,
        action_space=18,
        device=env.device,
    )

    dt = env.step_dt

    obs, _ = env.reset()
    step = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy.act(obs["policy"], hist_encoding=True)
            actions[:, 12:] = 0.0  # arm is IK-driven, keep only the 12 leg actions
            obs, rewards, terminated, truncated, info = env.step(actions)

        step += 1
        if step % args_cli.print_interval == 0:
            z = env.robot.data.root_pos_w[:, 2].mean().item()
            print(
                f"[play] step {step}: reward={rewards.mean().item():.4f} "
                f"term={terminated.sum().item()} trunc={truncated.sum().item()} "
                f"base_z={z:.3f}",
                flush=True,
            )

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
