# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ported v1-style RSL-RL stack used by the low-level policy.

Faithful port of the original ``visual_wholebody`` project's custom RSL-RL
modules (``third_party/rsl_rl``). Unlike the stock rsl_rl v3 that ships with
Isaac Lab, these modules implement the original algorithm design:

    - custom ActorCritic with prop/priv/hist observation split, dual control heads,
      a ``priv_encoder`` and a ``StateHistoryEncoder`` (CNN over history);
    - custom PPO with a dual reward channel (leg/arm), value mixing, a privileged
      regularizer loss (priv_reg), a minimum policy std and an internal DAgger
      (``update_dagger``) update that only trains the history encoder;
    - custom OnPolicyRunner that alternates ``hist_encoding`` sampling and runs
      ``update_dagger`` every ``dagger_update_freq`` iterations.

The ported modules are self-contained (no dependency on a specific rsl_rl version);
only ``torch`` is required. ``VisualWholeBodyVecEnvWrapper`` adapts an Isaac Lab
``DirectRLEnv`` to the plain-tensor v1 ``VecEnv`` interface these modules expect.
"""

from .actor_critic import ActorCritic, StateHistoryEncoder
from .config import get_b1z1_ppo_cfg
from .on_policy_runner import OnPolicyRunner
from .ppo import PPO
from .rollout_storage import RolloutStorage
from .vec_env import VisualWholeBodyVecEnvWrapper
