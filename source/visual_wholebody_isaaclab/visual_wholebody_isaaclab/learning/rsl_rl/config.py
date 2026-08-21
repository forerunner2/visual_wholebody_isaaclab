# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the low-level policy, aligned with the original project.

Faithful port of ``B1Z1RoughCfgPPO`` from ``visual_wholebody/low-level/legged_gym/envs/manip_loco/b1z1_config.py``.
It drives the custom v1-style RSL-RL stack (custom ActorCritic + custom PPO + custom
runner) under ``visual_wholebody_isaaclab.learning.rsl_rl``.
"""

from __future__ import annotations


def get_b1z1_ppo_cfg() -> dict:
    """Return the low-level PPO configuration dict (v1-style, original values)."""
    return {
        "seed": 1,
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": 45000,
            "save_interval": 200,
            "experiment_name": "b1z1_low_level_v1",
            "run_name": "",
            "resume": False,
            "load_run": -1,
            "checkpoint": -1,
        },
        "policy": {
            "continue_from_last_std": True,
            "init_std": [[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
            "actor_hidden_dims": [128],
            "critic_hidden_dims": [128],
            "activation": "elu",
            "output_tanh": False,
            "leg_control_head_hidden_dims": [128, 128],
            "arm_control_head_hidden_dims": [128, 128],
            "priv_encoder_dims": [64, 20],
            "num_leg_actions": 12,
            "num_arm_actions": 6,
            "adaptive_arm_gains": False,
            "adaptive_arm_gains_scale": 10.0,
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 2.0e-4,
            "schedule": "fixed",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": None,
            "max_grad_norm": 1.0,
            "min_policy_std": [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3],
            "mixing_schedule": [1.0, 0, 3000],
            "torque_supervision": False,
            "torque_supervision_schedule": [0.0, 1000, 1000],
            "adaptive_arm_gains": False,
            "dagger_update_freq": 20,
            "priv_reg_coef_schedual": [0, 0.1, 3000, 7000],
        },
    }
