# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from . import agents

##
# Register Gym environments.
##

import gymnasium as gym

from .pick_multi_env import VisualWholeBodyPickMultiEnv
from .pick_multi_env_cfg import VisualWholeBodyPickMultiEnvCfg
from .pick_multi_vision_env import VisualWholeBodyPickMultiVisionEnv
from .pick_multi_vision_env_cfg import VisualWholeBodyPickMultiVisionEnvCfg


gym.register(
    id="Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0",
    entry_point="visual_wholebody_isaaclab.tasks.direct.pick_multi.pick_multi_env:VisualWholeBodyPickMultiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VisualWholeBodyPickMultiEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0",
    entry_point="visual_wholebody_isaaclab.tasks.direct.pick_multi.pick_multi_vision_env:VisualWholeBodyPickMultiVisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": VisualWholeBodyPickMultiVisionEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
