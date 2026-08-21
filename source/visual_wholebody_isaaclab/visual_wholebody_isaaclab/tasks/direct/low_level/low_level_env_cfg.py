# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the B1+Z1 low-level whole-body control environment.

This is a DirectRLEnv port of ``ManipLoco`` from the original ``visual_wholebody``
Isaac Gym project. The time structure is preserved:

    physics dt = 0.005 s      (200 Hz)
    decimation = 4            -> policy dt = 0.02 s (50 Hz)
    episode_length_s = 10     -> 500 policy steps
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from visual_wholebody_isaaclab.assets.b1_z1_cfg import B1_Z1_CFG


@configclass
class GoalEECfg:
    """EE goal sampling (port of ``B1Z1RoughCfg.goal_ee``)."""

    num_commands: int = 3
    traj_time: list[float] = MISSING  # type: ignore
    hold_time: list[float] = MISSING  # type: ignore
    collision_upper_limits: list[float] = MISSING  # type: ignore
    collision_lower_limits: list[float] = MISSING  # type: ignore
    underground_limit: float = -0.7
    num_collision_check_samples: int = 10
    command_mode: str = "sphere"
    arm_induced_pitch: float = 0.38

    x_offset: float = 0.3
    y_offset: float = 0.0
    z_invariant_offset: float = 0.7

    init_pos_start: list[float] = MISSING  # type: ignore
    init_pos_end: list[float] = MISSING  # type: ignore
    pos_l: list[float] = MISSING  # type: ignore
    pos_p: list[float] = MISSING  # type: ignore
    pos_y: list[float] = MISSING  # type: ignore
    delta_orn_r: list[float] = MISSING  # type: ignore
    delta_orn_p: list[float] = MISSING  # type: ignore
    delta_orn_y: list[float] = MISSING  # type: ignore

    sphere_error_scale: list[float] = MISSING  # type: ignore
    orn_error_scale: list[float] = MISSING  # type: ignore


@configclass
class NoiseCfg:
    add_noise: bool = False
    noise_level: float = 1.0
    dof_pos: float = 0.01
    dof_vel: float = 1.5
    lin_vel: float = 0.1
    ang_vel: float = 0.2
    gravity: float = 0.05
    height_measurements: float = 0.1


@configclass
class CommandsCfg:
    curriculum: bool = True
    num_commands: int = 3
    resampling_time: float = 3.0
    lin_vel_x_schedule: list[float] = MISSING  # type: ignore
    ang_vel_yaw_schedule: list[float] = MISSING  # type: ignore
    tracking_ang_vel_yaw_schedule: list[float] = MISSING  # type: ignore
    ang_vel_yaw_clip: float = 0.5
    lin_vel_x_clip: float = 0.2
    lin_vel_x: list[float] = MISSING  # type: ignore
    ang_vel_yaw: list[float] = MISSING  # type: ignore


@configclass
class NormalizationCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    height_measurements: float = 5.0
    clip_observations: float = 100.0
    clip_actions: float = 100.0


@configclass
class InitStateCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.5)
    rand_yaw_range: float = np.pi / 2
    origin_perturb_range: float = 0.5
    init_vel_perturb_range: float = 0.1


@configclass
class ControlCfg:
    stiffness_joint: float = 80.0
    damping_joint: float = 2.0
    stiffness_z1: float = 5.0
    damping_z1: float = 0.5
    adaptive_arm_gains: bool = False
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale: list[float] = MISSING  # type: ignore
    decimation: int = 4
    torque_supervision: bool = False


@configclass
class DomainRandCfg:
    observe_priv: bool = True
    randomize_friction: bool = False  # not yet implemented; enable after phase-1 alignment
    friction_range: list[float] = MISSING  # type: ignore
    randomize_base_mass: bool = False
    added_mass_range: list[float] = MISSING  # type: ignore
    randomize_base_com: bool = False
    added_com_range_x: list[float] = MISSING  # type: ignore
    added_com_range_y: list[float] = MISSING  # type: ignore
    added_com_range_z: list[float] = MISSING  # type: ignore
    randomize_motor: bool = False
    leg_motor_strength_range: list[float] = MISSING  # type: ignore
    arm_motor_strength_range: list[float] = MISSING  # type: ignore
    randomize_gripper_mass: bool = False
    gripper_added_mass_range: list[float] = MISSING  # type: ignore
    push_robots: bool = False
    push_interval_s: float = 8.0
    max_push_vel_xy: float = 0.5


@configclass
class RewardsCfg:
    only_positive_rewards: bool = False
    tracking_sigma: float = 0.2
    tracking_ee_sigma: float = 1.0
    soft_dof_pos_limit: float = 1.0
    soft_dof_vel_limit: float = 1.0
    soft_torque_limit: float = 0.4
    base_height_target: float = 0.55
    max_contact_force: float = 40.0

    gait_vel_sigma: float = 0.5
    gait_force_sigma: float = 0.5
    kappa_gait_probs: float = 0.07
    feet_height_target: float = 0.3

    feet_aritime_allfeet: bool = False
    feet_height_allfeet: bool = False

    # leg reward scales (port of ``B1Z1RoughCfg.rewards.scales``)
    # gait rewards (active only when walking; the original defines delta_torques twice and the
    # second definition wins, so its effective value is -1.0e-7 -- see b1z1_config.py)
    feet_air_time: float = 2.0
    feet_height: float = 1.0
    tracking_lin_vel_max: float = 2.0
    tracking_ang_vel: float = 0.5
    delta_torques: float = -1.0e-7
    torques: float = -2.5e-5
    stand_still: float = 3.0
    # NOTE: scale raised 1.0 -> 3.0 (deliberate deviation from the original) so that, together
    # with the hardened exp(-dof_error*0.5) in `_reward_stand_still`, the "3-leg stance with one
    # leg raised" cheat is strictly worse than the default 4-leg stance. Diagnostic: the `work`
    # term saves ~0.0104/step by unloading a leg, and raising FL costs ~0.014/step of stand_still
    # at scale 3.0 -- the 4-leg stance is now clearly better. Tunable; reduce to 2.0 if the stance
    # looks too rigid after retraining.
    walking_dof: float = 1.5
    alive: float = 1.0
    lin_vel_z: float = -1.5
    roll: float = -2.0
    ang_vel_xy: float = -0.2
    dof_acc: float = -7.5e-7
    collision: float = -10.0
    action_rate: float = -0.015
    dof_pos_limits: float = -10.0
    hip_pos: float = -0.3
    work: float = -0.003
    feet_jerk: float = -0.0002
    feet_drag: float = -0.08
    feet_contact_forces: float = -0.001
    base_height: float = -5.0

    # arm reward scales (port of ``B1Z1RoughCfg.rewards.arm_scales``)
    tracking_ee_world: float = 0.8

    # log-only (zero scale) metrics
    tracking_ee_sphere: float = 0.0
    tracking_ee_orn: float = 0.0


@configclass
class TerminationCfg:
    r_threshold: float = 0.8
    p_threshold: float = 0.8
    z_threshold: float = 0.1


@configclass
class VisualWholeBodyLowLevelEnvCfg(DirectRLEnvCfg):
    """DirectRL environment config for the B1+Z1 low-level controller."""

    # RL
    action_space: int = 12 + 6
    observation_space: int = 744
    state_space: int = 0

    # time structure
    decimation: int = 4
    episode_length_s: float = 10.0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot_cfg: object = B1_Z1_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=0,
        debug_vis=False,
    )

    # joint / body groups
    num_gripper_joints: int = 1
    num_torques: int = 12 + 6
    num_proprio: int = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
    num_priv: int = 5 + 1 + 12
    history_len: int = 10
    action_delay_start_step: int = 10000 * 24  # original hardcoded ``10000 * 24`` (policy steps)
    action_delay: int = 3  # -1 to disable
    observe_gait_commands: bool = False
    stand_by: bool = False
    teleop_mode: bool = False
    stop_update_goal: bool = False
    record_video: bool = False
    # draw the EE-goal debug markers (goal / current EE / arm-base center), as in the original
    debug_viz: bool = False
    frequencies: int = 2

    # sub-configs
    goal_ee: GoalEECfg = MISSING  # type: ignore
    noise: NoiseCfg = MISSING  # type: ignore
    commands: CommandsCfg = MISSING  # type: ignore
    normalization: NormalizationCfg = MISSING  # type: ignore
    init_state: InitStateCfg = MISSING  # type: ignore
    control: ControlCfg = MISSING  # type: ignore
    domain_rand: DomainRandCfg = MISSING  # type: ignore
    rewards: RewardsCfg = MISSING  # type: ignore
    termination: TerminationCfg = MISSING  # type: ignore

    # arm / EE
    arm_base_offset: tuple[float, float, float] = (0.3, 0.0, 0.09)
    init_target_ee_base: tuple[float, float, float] = (0.2, 0.0, 0.2)
    grasp_offset: float = 0.08

    def __post_init__(self):
        """Fill in the nested defaults to keep the config concise."""
        # goal_ee
        self.goal_ee = GoalEECfg(
            traj_time=[1, 3],
            hold_time=[0.5, 2],
            collision_upper_limits=[0.1, 0.2, -0.05],
            collision_lower_limits=[-0.8, -0.2, -0.7],
            init_pos_start=[0.5, np.pi / 8, 0],
            init_pos_end=[0.7, 0, 0],
            pos_l=[0.4, 0.95],
            pos_p=[-1 * np.pi / 2.5, 1 * np.pi / 3],
            pos_y=[-1.2, 1.2],
            delta_orn_r=[-0.5, 0.5],
            delta_orn_p=[-0.5, 0.5],
            delta_orn_y=[-0.5, 0.5],
            sphere_error_scale=[1, 1, 1],
            orn_error_scale=[1, 1, 1],
        )
        # noise
        self.noise = NoiseCfg()
        # commands
        self.commands = CommandsCfg(
            lin_vel_x_schedule=[0, 0.5],
            ang_vel_yaw_schedule=[0, 1],
            tracking_ang_vel_yaw_schedule=[0, 1],
            lin_vel_x=[-0.8, 0.8],
            ang_vel_yaw=[-1.0, 1.0],
        )
        # normalization
        self.normalization = NormalizationCfg()
        # init state
        self.init_state = InitStateCfg()
        # control
        self.control = ControlCfg(
            action_scale=[0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        )
        # domain randomization
        self.domain_rand = DomainRandCfg(
            friction_range=[0.3, 3.0],
            added_mass_range=[0.0, 15.0],
            added_com_range_x=[-0.15, 0.15],
            added_com_range_y=[-0.15, 0.15],
            added_com_range_z=[-0.15, 0.15],
            leg_motor_strength_range=[0.7, 1.3],
            arm_motor_strength_range=[0.7, 1.3],
            gripper_added_mass_range=[0, 0.1],
        )
        # rewards
        self.rewards = RewardsCfg()
        # termination
        self.termination = TerminationCfg()

