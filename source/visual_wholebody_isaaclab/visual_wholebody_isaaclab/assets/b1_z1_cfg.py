# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree B1 quadruped with a Unitree Z1 arm (B1+Z1).

This articulation is the one used in the original ``visual_wholebody`` Isaac Gym
project. The active joints are 12 leg joints, 6 arm joints and 1 gripper joint.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Path to the low-level URDF (used for walking/locomotion tasks). Kept identical
# to the original ``visual_wholebody`` project.
B1_Z1_URDF_PATH = os.path.join(os.path.dirname(__file__), "data", "b1z1", "urdf", "b1z1.urdf")
# Path to the high-level "collection" URDF (used for grasping tasks, has simplified gripper collision).
B1_Z1_COL_URDF_PATH = os.path.join(os.path.dirname(__file__), "data", "b1z1-col", "urdf", "b1z1.urdf")

# Default joint positions from the original low-level config ``B1Z1RoughCfg.init_state.default_joint_angles``.
B1_Z1_DEFAULT_JOINT_POS = {
    "FR_hip_joint": -0.2,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "FL_hip_joint": 0.2,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RR_hip_joint": -0.2,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
    "RL_hip_joint": 0.2,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "z1_waist": 0.0,
    "z1_shoulder": 1.48,
    "z1_elbow": -0.63,
    "z1_wrist_angle": -0.84,
    "z1_forearm_roll": 0.0,
    "z1_wrist_rotate": 1.57,
    "z1_jointGripper": -0.785,
}

# Joint name groups (must be queried by name, never assumed by array order).
LEG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
ARM_JOINT_NAMES = [
    "z1_waist",
    "z1_shoulder",
    "z1_elbow",
    "z1_wrist_angle",
    "z1_forearm_roll",
    "z1_wrist_rotate",
]
GRIPPER_JOINT_NAME = "z1_jointGripper"

# Body / link names.
EE_BODY_NAME = "ee_gripper_link"
WRIST_BODY_NAME = "link06"
FOOT_BODY_NAME = "foot"
PENALIZED_CONTACT_NAMES = ["thigh", "trunk", "calf"]

# Training order of the 18 non-gripper joints used by the RL policy.
# Legs are ordered [FL, FR, RL, RR] (matching the original ``reindex_all``),
# followed by the 6 Z1 arm joints.
TRAIN_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle", "z1_forearm_roll", "z1_wrist_rotate",
]

# Joint effort limits (N*m) keyed by joint name, from the original URDF.
JOINT_EFFORT_LIMITS = {
    "FR_hip_joint": 91.0035, "FR_thigh_joint": 93.33, "FR_calf_joint": 140.0,
    "FL_hip_joint": 91.0035, "FL_thigh_joint": 93.33, "FL_calf_joint": 140.0,
    "RR_hip_joint": 91.0035, "RR_thigh_joint": 93.33, "RR_calf_joint": 140.0,
    "RL_hip_joint": 91.0035, "RL_thigh_joint": 93.33, "RL_calf_joint": 140.0,
    "z1_waist": 30.0, "z1_shoulder": 60.0, "z1_elbow": 30.0,
    "z1_wrist_angle": 30.0, "z1_forearm_roll": 30.0, "z1_wrist_rotate": 30.0,
    "z1_jointGripper": 30.0,
}


def _b1_z1_urdf_file_cfg(asset_path: str) -> sim_utils.UrdfFileCfg:
    """Build the shared ``UrdfFileCfg`` for the B1+Z1 robot.

    The legs are driven by an explicit PD controller written with
    :meth:`set_joint_effort_target` (matching the original ``_compute_torques``),
    so the simulation-level joint gains for the legs are set to zero to avoid a
    double PD. The arm and gripper are driven by position targets.
    """
    return sim_utils.UrdfFileCfg(
        asset_path=asset_path,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
    )


# Low-level B1+Z1 used by the low-level walking environment.
B1_Z1_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=_b1_z1_urdf_file_cfg(B1_Z1_URDF_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos=B1_Z1_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            effort_limit_sim=140.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim=30.0,
            velocity_limit_sim=6.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT_NAME],
            effort_limit_sim=20.0,
            velocity_limit_sim=2.0,
            stiffness=40.0,
            damping=2.5,
        ),
    },
)

# High-level B1+Z1 (collection URDF) used by the pick-multi grasping environment.
B1_Z1_COL_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=_b1_z1_urdf_file_cfg(B1_Z1_COL_URDF_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos=B1_Z1_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            effort_limit_sim=140.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim=30.0,
            velocity_limit_sim=6.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[GRIPPER_JOINT_NAME],
            effort_limit_sim=20.0,
            velocity_limit_sim=2.0,
            stiffness=40.0,
            damping=2.5,
        ),
    },
)
