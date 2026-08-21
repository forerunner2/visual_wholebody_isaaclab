# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the B1+Z1 high-level multi-object pick environment (state teacher).

Port of ``B1Z1PickMulti`` from the original ``visual_wholebody`` project. The
high-level policy runs at ~6.25 Hz (0.16 s), the frozen low-level policy at 50 Hz
and the physics at 200 Hz:

    physics dt = 0.005 s
    low-level decimation = 4            (physics steps per low-level step)
    low-level steps per high-level step = 8
    high-level decimation = 32          (physics steps per high-level step)
    episode_length_s = 150 * 0.16 = 24 s
"""

from __future__ import annotations

import copy
import os

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from visual_wholebody_isaaclab.assets.b1_z1_cfg import B1_Z1_COL_CFG

# object assets used by the first version (geometrically distinct objects)
OBJ_SET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets", "data", "obj_set")


def _obj_urdf(name: str) -> str:
    return os.path.join(OBJ_SET_DIR, name, "model.urdf")


@configclass
class PickMultiRewardsCfg:
    only_positive_rewards: bool = False
    base_height_target: float = 0.55

    approaching: float = 0.5
    lifting: float = 1.0
    pick_up: float = 3.5
    acc_penalty: float = -0.001
    command_penalty: float = -1.0
    command_reward: float = 0.25
    standpick: float = 0.25
    action_rate: float = -0.001
    ee_orn: float = 0.01
    base_dir: float = 0.25
    base_approaching: float = 0.01
    grasp_base_height: float = 0.5


@configclass
class PickMultiObjectCfg:
    """Parameters for one graspable object."""

    name: str = MISSING  # type: ignore
    height: float = MISSING  # type: ignore
    orientation: tuple[float, float, float, float] = MISSING  # type: ignore
    scale: float = 1.0
    feature_path: str = MISSING  # type: ignore


@configclass
class VisualWholeBodyPickMultiEnvCfg(DirectRLEnvCfg):
    """DirectRL environment config for the B1+Z1 state teacher grasp task."""

    # RL
    action_space: int = 9
    state_space: int = 0

    # time structure
    decimation: int = 32
    episode_length_s: float = 24.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)

    # assets
    robot_cfg: object = B1_Z1_COL_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.125)),
    )
    # single-object first version; extend to a RigidObjectCollection for 3+ objects
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UrdfFileCfg(asset_path=_obj_urdf("sugar_box"), fix_base=False, joint_drive=None),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.338)),
    )

    # environment settings
    lifted_success_threshold: float = 0.35
    lifted_init_threshold: float = 0.05
    base_object_dist_threshold: float = 0.6
    hold_steps: int = 25
    last_commands: bool = False
    use_tanh: bool = False
    near_goal_stop: bool = False
    obj_move_prob: float = 0.0
    commands_curriculum: bool = True
    pitch_control: bool = False
    stop_pick: bool = False
    rand_control: bool = False
    arm_delay: bool = False
    small_value_set_zero: bool = False

    # low-level policy (trained in Isaac Lab)
    # NOTE: update to the final low-level checkpoint once low-level training completes.
    low_policy_path: str = "/home/haotian/workspace/visual_wholebody_isaaclab/logs/rsl_rl/b1z1_low_level_v1/2026-08-21_09-55-47/model_15000.pt"
    low_level_obs_space: int = 726  # 66 current + 66*10 history (no gait)
    low_level_action_space: int = 18
    observe_gait_commands: bool = False

    # object features
    no_feature: bool = False

    # reward
    rewards: PickMultiRewardsCfg = MISSING  # type: ignore

    # obs / action dims
    num_proprio: int = 66
    num_gripper_joints: int = 1
    num_actions: int = 9

    def __post_init__(self):
        self.rewards = PickMultiRewardsCfg()
        # 机器人 2 m 外起步（原版训练 robot_start_pose=(-2.0,0,0.55)），避免站在桌子上穿模；
        # 深拷贝一份，防止污染共享的 B1_Z1_COL_CFG（低层 env 也在用）
        self.robot_cfg = copy.deepcopy(B1_Z1_COL_CFG)
        self.robot_cfg.init_state.pos = (-2.0, 0.0, 0.55)
        if self.no_feature:
            self.observation_space = 70
        else:
            self.observation_space = 70 + 1024  # 38-1 + 9 + 24 + feature(1024)
