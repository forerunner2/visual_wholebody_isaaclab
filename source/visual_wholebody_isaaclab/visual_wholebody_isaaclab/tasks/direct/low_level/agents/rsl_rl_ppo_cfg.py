# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the B1+Z1 low-level environment.

.. note::
    The low-level policy is trained with the ORIGINAL custom RSL-RL stack (custom
    ActorCritic / PPO / runner with prop/priv/hist split, dual heads, priv_encoder,
    StateHistoryEncoder, priv_reg distillation and the internal DAgger loop). See
    ``scripts/rsl_rl/train_v1.py`` and ``visual_wholebody_isaaclab.learning.rsl_rl``.

    This file keeps a stock ``RslRlOnPolicyRunnerCfg`` only as a fallback/reference
    for the standard Isaac Lab v3 workflow. It is NOT what the low-level policy is
    trained with.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 45000
    save_interval = 200
    experiment_name = "b1z1_low_level"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=None,
        max_grad_norm=1.0,
    )
