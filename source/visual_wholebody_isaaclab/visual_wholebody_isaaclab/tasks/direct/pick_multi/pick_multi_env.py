# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for the B1+Z1 high-level multi-object pick task (state teacher).

Direct port of ``B1Z1PickMulti`` from the original ``visual_wholebody`` Isaac Gym
project. The high-level policy produces a 9-dim action:

    [0:3]  EE position delta   (clipped to +-0.02 m)
    [3:6]  EE RPY delta        (clipped to +-0.06 rad)
    [6]    gripper command     (>=0 open, <0 close)
    [7]    forward velocity    (curriculum)
    [8]    yaw velocity        (clipped to +-0.6 rad/s)

The environment internally runs the frozen low-level policy at 50 Hz.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_conjugate, quat_from_euler_xyz, quat_mul, quat_rotate_inverse

from visual_wholebody_isaaclab.assets.b1_z1_cfg import (
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    FOOT_BODY_NAME,
    GRIPPER_JOINT_NAME,
    JOINT_EFFORT_LIMITS,
    LEG_JOINT_NAMES,
    TRAIN_JOINT_NAMES,
)
from visual_wholebody_isaaclab.learning.low_level_policy import LowLevelPolicy
from visual_wholebody_isaaclab.utils.math_utils import orientation_error, torch_rand_float

from .pick_multi_env_cfg import VisualWholeBodyPickMultiEnvCfg

LIN_VEL_X_CLIP = 0.15
ANG_VEL_YAW_CLIP = 0.35
ANG_VEL_PITCH_CLIP = 0.35


class VisualWholeBodyPickMultiEnv(DirectRLEnv):
    """High-level B1+Z1 multi-object pick environment (state teacher)."""

    cfg: VisualWholeBodyPickMultiEnvCfg

    def __init__(self, cfg: VisualWholeBodyPickMultiEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # resolve joints / bodies
        self.leg_joint_ids, _ = self.robot.find_joints(LEG_JOINT_NAMES)
        self.arm_joint_ids, _ = self.robot.find_joints(ARM_JOINT_NAMES)
        self.gripper_joint_ids, _ = self.robot.find_joints(GRIPPER_JOINT_NAME)
        self.ee_body_id = self.robot.find_bodies(EE_BODY_NAME)[0][0]
        self.wrist_body_id = self.robot.find_bodies("link06")[0][0]
        self.foot_body_ids, self.foot_body_names = self.robot.find_bodies(f".*{FOOT_BODY_NAME}.*")
        # NOTE: articulation bodies and contact-sensor bodies live in two *different*
        # index spaces (``robot.data.*`` / ``robot.find_bodies`` vs
        # ``contact_sensor.data.net_forces_w`` / ``contact_sensor.find_bodies``). The
        # low-level teacher policy reads ``foot_contacts`` from ``net_forces_w``, so it
        # must be sliced with the *sensor* ids below, never with ``foot_body_ids``.
        self.foot_contact_ids, self.foot_contact_names = self.contact_sensor.find_bodies(f".*{FOOT_BODY_NAME}.*")

        # training<->robot joint permutation (same convention as the low-level env).
        # The low-level policy consumes/produces joints in TRAIN_JOINT_NAMES order;
        # the simulator exposes joints in USD-tree (interleaved) order.
        self.train_joint_names = list(TRAIN_JOINT_NAMES)
        self.perm = torch.tensor(
            [self.robot.joint_names.index(n) for n in self.train_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        # full 19-joint permutation: training order followed by the gripper
        self.perm_full = torch.tensor(
            [self.robot.joint_names.index(n) for n in self.train_joint_names + [GRIPPER_JOINT_NAME]],
            dtype=torch.long,
            device=self.device,
        )
        # robot-order indices of the 12 leg / 6 arm joints
        self.leg_indices = torch.tensor(self.leg_joint_ids, dtype=torch.long, device=self.device)
        self.arm_indices = torch.tensor(self.arm_joint_ids, dtype=torch.long, device=self.device)
        # feet contact permutation -> canonical [FL, FR, RL, RR] *in the sensor index
        # space*. ``foot_contacts_from_sensor`` is built from ``net_forces_w`` in
        # ``_update_derived_state`` and is already canonical, so the low-level observation
        # concatenates it directly (no further reordering).
        foot_order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.feet_contact_perm = torch.tensor(
            [self.foot_contact_names.index(n) for n in foot_order], dtype=torch.long, device=self.device
        )

        self.num_gripper_joints = self.cfg.num_gripper_joints
        self.dt = self.cfg.decimation * self.cfg.sim.dt  # high-level dt = 0.16 s
        self.low_level_dt = 4 * self.cfg.sim.dt  # 0.02 s
        # NOTE: max_episode_length is a read-only property provided by DirectRLEnv
        self.num_actions = self.cfg.action_space
        self.clip_actions = 100.0
        self.clip_obs = 100.0

        # low-level policy
        self.low_level_policy = LowLevelPolicy(
            checkpoint_path=self.cfg.low_policy_path,
            obs_space=self.cfg.low_level_obs_space,
            action_space=self.cfg.low_level_action_space,
            device=self.device,
        )

        # buffers
        self._init_buffers()

        # reward container
        self.reward_scales = {
            "approaching": self.cfg.rewards.approaching,
            "lifting": self.cfg.rewards.lifting,
            "pick_up": self.cfg.rewards.pick_up,
            "acc_penalty": self.cfg.rewards.acc_penalty,
            "command_penalty": self.cfg.rewards.command_penalty,
            "command_reward": self.cfg.rewards.command_reward,
            "standpick": self.cfg.rewards.standpick,
            "action_rate": self.cfg.rewards.action_rate,
            "ee_orn": self.cfg.rewards.ee_orn,
            "base_dir": self.cfg.rewards.base_dir,
            "base_approaching": self.cfg.rewards.base_approaching,
            "grasp_base_height": self.cfg.rewards.grasp_base_height,
        }
        self.reward_scales = {k: v for k, v in self.reward_scales.items() if v is not None and v != 0.0}
        self.reward_names = list(self.reward_scales.keys())
        self.reward_functions = [getattr(self, f"_reward_{n}") for n in self.reward_names]
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.reward_scales
        }
        self.episode_metric_sums = dict(self.episode_sums)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ steps

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.physics_substep = 0
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        # action delay (one step after 15000 env steps, as in the original)
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], action_tensor[:, None, :]], dim=1)
        if self.common_step_counter > 15000:
            action_delay = 1
        else:
            action_delay = 0
        action_tensor = self.action_history_buf[:, -action_delay - 1]
        if self.cfg.arm_delay and action_delay > 0:
            action_tensor[:, :7] = self.action_history_buf[:, -action_delay - 2, :7]
        self.actions[:] = action_tensor[:]

        # safety override (near_goal_stop): when close to the object, force the velocity
        # commands to zero and expose the corrected action as ``replaced_action`` so the
        # DAgger trainer can learn the safe labels (original ``B1Z1PickMulti.step``).
        if self.cfg.near_goal_stop:
            base_obj_dis = torch.norm(self.object_state[:, :2] - self.arm_base[:, :2], dim=-1)
            replaced_action = self.actions.clone()
            replaced_action[base_obj_dis < 0.6, 7:9] = 0.0
            self.extras["replaced_action"] = replaced_action
            self.actions[:] = replaced_action

        # ee goal position delta (in robot/arm-base frame)
        if self.cfg.use_tanh:
            self.delta_goal_cart = self.actions[:, :3] * 0.02
        else:
            self.delta_goal_cart = torch.clip(self.actions[:, :3], -0.02, 0.02)
        self.curr_ee_goal_cart = self.curr_ee_goal_cart + self.delta_goal_cart
        self.curr_ee_goal_cart[:, 0] = torch.clip(self.curr_ee_goal_cart[:, 0], 0.0, 0.7)
        self.curr_ee_goal_cart[:, 1] = torch.clip(self.curr_ee_goal_cart[:, 1], -0.7, 0.7)
        self.curr_ee_goal_cart[:, 2] = torch.clip(self.curr_ee_goal_cart[:, 2], -0.6, 0.6)
        self.ee_goal_cart_world = quat_apply(self.robot.data.root_quat_w, self.curr_ee_goal_cart) + self.arm_base

        # ee goal orientation delta
        if self.cfg.use_tanh:
            self.delta_goal_orn = self.actions[:, 3:6] * 0.06
        else:
            self.delta_goal_orn = torch.clip(self.actions[:, 3:6], -0.06, 0.06)
        self.curr_ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy + self.delta_goal_orn
        ee_goal_local_orn = quat_from_euler_xyz(
            self.curr_ee_goal_orn_rpy[:, 0], self.curr_ee_goal_orn_rpy[:, 1], self.curr_ee_goal_orn_rpy[:, 2]
        )
        self.ee_goal_orn_quat = quat_mul(self.robot.data.root_quat_w, ee_goal_local_orn)

        # gripper
        self._set_gripper()
        # base commands
        self._clip_commands()
        self.commands[:, 1] = 0.0
        if self.cfg.use_tanh:
            self.commands[:, 2] = self.actions[:, 8] * 0.6
        else:
            self.commands[:, 2] = torch.clip(self.actions[:, 8], -0.6, 0.6)
        if self.cfg.small_value_set_zero:
            self.commands *= (
                torch.logical_or(
                    torch.abs(self.commands[:, 0]) > LIN_VEL_X_CLIP,
                    torch.abs(self.commands[:, 2]) > ANG_VEL_YAW_CLIP,
                )
            ).unsqueeze(1)

        # store the effective (clipped) action for the action-history observation
        self.clipped_actions[:, :3] = self.delta_goal_cart
        self.clipped_actions[:, 3:6] = self.delta_goal_orn
        self.clipped_actions[:, 6] = self.actions[:, 6]
        self.clipped_actions[:, 7] = self.commands[:, 0]
        self.clipped_actions[:, 8] = self.commands[:, 2]
        self.command_history_buf = torch.cat([self.command_history_buf[:, 1:], self.clipped_actions[:, None, :]], dim=1)

        if self.cfg.stop_pick:
            self.commands = torch.where(
                self.actions[:, 6].unsqueeze(-1) < 0, torch.zeros_like(self.commands), self.commands
            )

    def _apply_action(self) -> None:
        if self.physics_substep % 4 == 0:
            # run frozen low-level policy (50 Hz); output is in training order
            low_obs = self._compute_low_level_observations()
            low_actions_train = self.low_level_policy.act(low_obs.detach()).clone()
            low_actions_train[:, 12:] = 0.0  # arm is IK-driven, not policy-driven
            # scatter training -> robot order for the torque computation
            low_actions_robot = torch.zeros_like(low_actions_train)
            low_actions_robot[:, self.perm] = low_actions_train
            self.low_level_actions = low_actions_robot
            self.last_low_actions[:] = self.low_level_actions[:]

        # legs: explicit PD torques
        self.torques = self._compute_torques(self.last_low_actions)
        self.robot.set_joint_effort_target(self.torques, joint_ids=None)
        # arm: differential IK position targets
        dpos = self.ee_goal_cart_world - self.ee_pos
        ee_orn = self.robot.data.body_quat_w[:, self.ee_body_id]
        ee_orn = ee_orn / torch.norm(ee_orn, dim=-1).unsqueeze(-1)
        drot = orientation_error(self.ee_goal_orn_quat, ee_orn)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        arm_pos_targets = self._control_ik(dpose) + self.robot.data.joint_pos[:, self.arm_joint_ids]
        self.robot.set_joint_position_target(arm_pos_targets, joint_ids=self.arm_joint_ids)
        # gripper position target
        self.robot.set_joint_position_target(self.gripper_dof_pos, joint_ids=self.gripper_joint_ids)

        self.physics_substep += 1

    # ------------------------------------------------------------------ observations

    def _get_observations(self) -> dict:
        self._update_derived_state()
        robot_obs = self._compute_robot_observations()
        if self.cfg.last_commands:
            obs = torch.cat([robot_obs, self.command_history_buf[:, -1]], dim=-1)
        else:
            obs = torch.cat([robot_obs, self.action_history_buf[:, -1]], dim=-1)
        if not self.cfg.no_feature:
            obs = torch.cat([self.feature_obs, obs], dim=-1)
        self.obs_buf = torch.clamp(obs, -self.clip_obs, self.clip_obs)

        # update last-step buffers (once per step, as in the low-level env). ``_reward_action_rate``
        # and ``_reward_acc_penalty`` compare against these on the next step; without this update
        # they only ever see zeros and penalize magnitude/velocity instead of change/acceleration.
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.robot.data.joint_vel[:]

        return {"policy": self.obs_buf}

    def _compute_robot_observations(self) -> torch.Tensor:
        robot_root_state = self.robot.data.root_state_w  # (N, 13)
        cube_root_state = self.object_state  # (N, 13)
        body_pos = self.robot.data.body_pos_w
        body_rot = self.robot.data.body_quat_w
        body_vel = self.robot.data.body_lin_vel_w
        body_ang_vel = self.robot.data.body_ang_vel_w
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        commands = self.commands
        base_quat_yaw = self.base_yaw_quat
        spherical_center = self.get_ee_goal_spherical_center()
        ee_goal_cart = self.curr_ee_goal_cart
        ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy

        cube_pos = cube_root_state[:, :3]
        cube_orn = cube_root_state[:, 3:7]

        ee_pos = body_pos[:, self.ee_body_id, :]
        ee_rot = body_rot[:, self.ee_body_id, :]
        ee_vel = body_vel[:, self.ee_body_id, :]
        ee_ang_vel = body_ang_vel[:, self.ee_body_id, :]

        dof_pos_r = dof_pos[:, self.perm_full]
        dof_vel_r = dof_vel[:, self.perm] * 0.05

        base_quat = robot_root_state[:, 3:7]
        arm_base = self.arm_base

        cube_pos_local = quat_rotate_inverse(base_quat_yaw, cube_pos - arm_base)
        cube_pos_local[:, 2] = cube_pos[:, 2]
        cube_orn_local = quat_mul(quat_conjugate(base_quat_yaw), cube_orn)
        roll, pitch, yaw = euler_xyz_from_quat(cube_orn_local)
        cube_orn_local_rpy = torch.stack([roll, pitch, yaw], dim=-1)

        ee_pos_local = quat_rotate_inverse(base_quat, ee_pos - arm_base)
        ee_rot_local = quat_mul(quat_conjugate(base_quat), ee_rot)
        eroll, epitch, eyaw = euler_xyz_from_quat(ee_rot_local)
        ee_rot_local_rpy = torch.stack([eroll, epitch, eyaw], dim=-1)

        robot_vel_local = quat_rotate_inverse(base_quat_yaw, robot_root_state[:, 7:10])

        obs = torch.cat(
            (
                cube_pos_local,
                cube_orn_local_rpy,
                ee_pos_local,
                ee_rot_local_rpy,
                dof_pos_r,
                dof_vel_r,
                commands,
                ee_goal_cart,
                ee_goal_orn_rpy,
                robot_vel_local,
            ),
            dim=-1,
        )
        return obs

    def _compute_low_level_observations(self) -> torch.Tensor:
        self._step_contact_targets()
        base_ang_vel = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.root_ang_vel_w)
        commands = self.commands.clone()
        low_obs = torch.cat(
            (
                self.get_body_orientation(),  # dim 2
                base_ang_vel,  # dim 3
                (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self.perm],  # 18
                self.robot.data.joint_vel[:, self.perm] * 0.05,  # 18
                self.last_low_actions[:, self.perm][:, :12],  # 12 (training-order legs)
                self.foot_contacts_from_sensor,  # 4, already canonical [FL, FR, RL, RR]
                commands[:, :3],  # 3
                self.curr_ee_goal_cart,  # 3
                0.0 * self.curr_ee_goal_cart,  # 3
            ),
            dim=-1,
        )
        if self.cfg.observe_gait_commands:
            low_obs = torch.cat((low_obs, self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)

        self.low_obs_history_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None],
            torch.stack([low_obs] * 10, dim=1),
            self.low_obs_history_buf,
        )
        self.low_obs_buf = torch.cat([low_obs, self.low_obs_history_buf.view(self.num_envs, -1)], dim=-1)
        self.low_obs_history_buf = torch.cat([self.low_obs_history_buf[:, 1:], low_obs.unsqueeze(1)], dim=1)
        return self.low_obs_buf

    # ------------------------------------------------------------------ rewards

    def _get_rewards(self) -> torch.Tensor:
        self.rew_buf[:] = 0.0
        for i, name in enumerate(self.reward_names):
            rew, metric = self.reward_functions[i]()
            rew = rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        return self.rew_buf

    def _reward_approaching(self):
        dist_delta = self.closest_dist - self.curr_dist
        self.closest_dist = torch.minimum(self.closest_dist, self.curr_dist)
        dist_delta = torch.clip(dist_delta, 0.0, 10.0)
        reward = torch.tanh(10.0 * dist_delta)
        reward = reward * (~self.lifted_object).float()
        return reward, reward

    def _reward_lifting(self):
        height_delta = self.curr_height - self.highest_object
        self.highest_object = torch.maximum(self.highest_object, self.curr_height)
        height_delta = torch.clip(height_delta, 0.0, 10.0)
        lifting_rew = torch.tanh(10.0 * height_delta)
        reward = torch.where(self.lifted_object, torch.zeros_like(lifting_rew), lifting_rew)
        return reward, reward

    def _reward_pick_up(self):
        reward = torch.where(
            self.lifted_object, torch.ones_like(self.reset_terminated, dtype=torch.float), torch.zeros_like(self.reset_terminated, dtype=torch.float)
        )
        if self.common_step_counter < 20000 or self.eval_mode:
            self.success_counter[self.lifted_object] += 1
        else:
            self.success_counter[self.lifted_object & (self.pick_counter < 1)] += 1
            self.pick_counter = torch.where(self.lifted_object, self.pick_counter + 1, torch.zeros_like(self.pick_counter))
        return reward, reward

    def _reward_acc_penalty(self):
        arm_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]
        last_arm_vel = self.last_dof_vel[:, self.arm_joint_ids]
        penalty = torch.norm(arm_vel - last_arm_vel, dim=-1) / self.dt
        return 1 - torch.exp(-penalty), 1 - torch.exp(-penalty)

    def _reward_command_reward(self):
        base_obj_dis = self.base_obj_dis
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        reward[base_obj_dis < 0.6] = torch.exp(-torch.abs(self.commands[:, 0]))[base_obj_dis < 0.6]
        if self.common_step_counter < 30000:
            reward = 0.0
        return reward, reward

    def _reward_command_penalty(self):
        base_obj_dis = self.base_obj_dis
        penalty = torch.where(
            base_obj_dis < 0.6,
            torch.norm(self.commands[:, :1], dim=-1),
            torch.zeros_like(self.reset_terminated, device=self.device, dtype=torch.float),
        )
        if self.common_step_counter < 30000:
            penalty = 0.0
        return penalty, penalty

    def _reward_standpick(self):
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        reward[(self.base_obj_dis < self.cfg.base_object_dist_threshold) & (self.commands[:, 0] < LIN_VEL_X_CLIP)] = 1.0
        if self.common_step_counter < 30000:
            reward = 0.0
        return reward, reward

    def _reward_ee_orn(self):
        ee_x_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        ee_x_dir_world = quat_apply(self.ee_orn, ee_x_dir)
        obj_dir = self.object_state[:, :3] - self.ee_pos
        obj_dist = torch.norm(obj_dir, dim=-1)
        far_obj = obj_dist >= 0.01
        rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        obj_dir_unit = obj_dir[far_obj] / obj_dist[far_obj].unsqueeze(-1)
        rew[far_obj] = torch.nn.functional.cosine_similarity(ee_x_dir_world[far_obj], obj_dir_unit)
        return rew, rew

    def _reward_base_dir(self):
        base_x_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        base_x_dir_world = quat_apply(self.base_yaw_quat, base_x_dir)
        obj_dir = self.object_state[:, :3] - self.robot.data.root_pos_w
        obj_dir[:, :2] = 0.0
        obj_dist = torch.norm(obj_dir, dim=-1)
        safe_dis = obj_dist >= 0.01
        rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        obj_dir_unit = obj_dir[safe_dis] / obj_dist[safe_dis].unsqueeze(-1)
        rew[safe_dis] = torch.nn.functional.cosine_similarity(base_x_dir_world[safe_dis], obj_dir_unit)
        return rew, rew

    def _reward_base_approaching(self):
        base_obj_dis = self.base_obj_dis
        delta_dis = torch.abs(base_obj_dis - self.cfg.base_object_dist_threshold)
        reward = torch.tanh(-10 * delta_dis) + 1
        return reward, reward

    def _reward_grasp_base_height(self):
        reward, _ = self._reward_base_height()
        reward = reward * self.lifted_now.float()
        return reward, reward

    def _reward_base_height(self):
        base_height = torch.mean(self.robot.data.root_pos_w[:, 2].unsqueeze(1), dim=1)
        reward = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target))
        return reward, base_height

    def _reward_action_rate(self):
        diff = torch.norm(self.actions[:, 7:9] - self.last_actions[:, 7:9], dim=-1)
        return diff, diff

    # ------------------------------------------------------------------ termination

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # refresh derived state right after physics. ``_get_dones`` runs before
        # ``_get_rewards``, and both depend on ee/object/foot state that would
        # otherwise lag one full step behind (the original refreshes these tensors
        # every step via ``_refresh_sim_tensors``).
        self._update_derived_state()
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        z = self.robot.data.root_pos_w[:, 2]
        r_term = torch.abs(roll) > 0.8
        p_term = torch.abs(pitch) > 0.8
        z_term = z < 0.1

        curr_ee_pos_local = quat_rotate_inverse(self.robot.data.root_quat_w, self.ee_pos - self.arm_base)
        ik_fail = (self.curr_ee_goal_cart[:, -1:] - curr_ee_pos_local[:, -1:]).norm(dim=-1) > 0.2

        cube_height = self.object_state[:, 2]
        d1 = torch.norm(self.object_state[:, :3] - self.ee_pos, dim=-1)
        self.lifted_now = torch.logical_and(
            (cube_height - self.table_height) > (0.03 / 2 + self.cfg.lifted_success_threshold), d1 < 0.1
        )
        self.lifted_object = torch.logical_and(
            (cube_height - self.table_height - self.obj_height) > (self.cfg.lifted_success_threshold), d1 < 0.1
        )

        cube_falls = cube_height < self.table_height

        # reset the episode once the object is picked (before 20000 steps) or held for
        # ``hold_steps`` (after) — port of the original ``_reward_pick_up`` reset_buf
        # side effect, which the Isaac Lab reset flow can only express via termination.
        if self.common_step_counter < 20000 or self.eval_mode:
            pick_completed = self.lifted_object
        else:
            pick_completed = self.pick_counter >= self.cfg.hold_steps

        terminated = r_term | p_term | z_term | ik_fail | cube_falls | pick_completed

        # dropped after lift -> terminate
        dropped = (~self.lifted_now) & self.lifted_object
        terminated = terminated | dropped

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.last_ee_pos = curr_ee_pos_local
        return terminated, time_out

    # ------------------------------------------------------------------ reset

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # robot root
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :2] += torch_rand_float(-0.2, 0.2, (len(env_ids), 2), self.device)
        rand_yaw = torch_rand_float(-0.8, 0.8, (len(env_ids), 1), self.device).squeeze(1)
        default_root_state[:, 3:7] = quat_from_euler_xyz(0.0 * rand_yaw, 0.0 * rand_yaw, rand_yaw)
        default_root_state[:, 7:13] = 0.0
        self.robot.data.root_pos_w[env_ids] = default_root_state[:, :3]
        self.robot.data.root_quat_w[env_ids] = default_root_state[:, 3:7]
        self.robot.data.root_lin_vel_w[env_ids] = default_root_state[:, 7:10]
        self.robot.data.root_ang_vel_w[env_ids] = default_root_state[:, 10:13]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # robot joints (reset to default + gripper randomization)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.gripper_joint_ids] += torch_rand_float(-0.5, 0.5, (len(env_ids), 1), self.device)
        self.robot.data.joint_pos[env_ids] = joint_pos
        self.robot.data.joint_vel[env_ids] = joint_vel
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # table (static; only used to set the table-top height)
        self.table_height[env_ids] = self.table_z_center + self.table_half_height

        # object
        obj_state = torch.zeros(len(env_ids), 13, device=self.device)
        obj_state[:, 0] = torch_rand_float(-0.15, 0.15, (len(env_ids), 1), self.device).squeeze(1)
        obj_state[:, 1] = torch_rand_float(-0.1, 0.1, (len(env_ids), 1), self.device).squeeze(1)
        obj_state[:, 2] = self.table_height[env_ids] + self.obj_height[env_ids]
        rand_yaw_box = torch_rand_float(-3.15, 3.15, (len(env_ids), 1), self.device).squeeze(1)
        init_orn = self.obj_orn_tensor  # (4,) wxyz
        yaw_quat = quat_from_euler_xyz(0.0 * rand_yaw_box, 0.0 * rand_yaw_box, rand_yaw_box)
        obj_state[:, 3:7] = quat_mul(yaw_quat, init_orn.repeat(len(env_ids), 1))
        obj_state[:, 7:13] = 0.0
        self.object.write_root_pose_to_sim(obj_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj_state[:, 7:], env_ids)
        self.object_state[env_ids] = obj_state
        self.object_root_state[env_ids] = obj_state

        # refresh buffers
        self.scene.update(dt=self.physics_dt)
        # recompute arm_base / ee_pos / ee_orn / object_state from the freshly-written
        # reset pose so the ee-goal world transform below is not stale
        self._update_derived_state()

        # buffers
        self.last_actions[env_ids] = 0.0
        self.last_low_actions[env_ids] = 0.0
        self.clipped_actions[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self.action_history_buf[env_ids] = 0.0
        self.command_history_buf[env_ids] = 0.0
        self.curr_dist[env_ids] = 0.0
        self.closest_dist[env_ids] = -1.0
        self.reach_counter[env_ids] = 0
        self.pick_counter[env_ids] = 0
        self.lifted_object[env_ids] = 0
        self.lifted_now[env_ids] = 0
        self.curr_height[env_ids] = 0.0
        self.highest_object[env_ids] = -1.0
        self.low_obs_history_buf[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

        # reset ee goal to default
        self.curr_ee_goal_cart[env_ids] = self.init_ee_goal_cart[env_ids]
        self.curr_ee_goal_orn_rpy[env_ids, :] = torch.tensor([np.pi / 2, 0.0, 0.0], device=self.device)
        self.ee_goal_cart_world[env_ids] = (
            quat_apply(self.robot.data.root_quat_w[env_ids], self.curr_ee_goal_cart[env_ids]) + self.arm_base[env_ids]
        )
        self.ee_goal_orn_quat[env_ids] = quat_mul(
            self.robot.data.root_quat_w[env_ids],
            quat_from_euler_xyz(
                self.curr_ee_goal_orn_rpy[env_ids, 0],
                self.curr_ee_goal_orn_rpy[env_ids, 1],
                self.curr_ee_goal_orn_rpy[env_ids, 2],
            ),
        )

    # ------------------------------------------------------------------ helpers

    def _init_buffers(self):
        self.arm_base_offset = torch.tensor([0.3, 0.0, 0.09], device=self.device).repeat(self.num_envs, 1)
        self.ee_goal_center_offset = torch.tensor([0.3, 0.0, 0.7], device=self.device).repeat(self.num_envs, 1)
        self.table_z_center = 0.125
        # 原版 table_heights 语义 = 桌面顶面高度 = 中心 z + 半高 (0.125 + 0.125 = 0.25)
        self.table_half_height = self.cfg.table_cfg.spawn.size[2] / 2.0
        self.table_height = torch.full(
            (self.num_envs,), self.table_z_center + self.table_half_height, device=self.device, dtype=torch.float
        )
        self.obj_height = torch.full((self.num_envs,), 0.088, device=self.device, dtype=torch.float)  # sugar_box
        self.obj_orn_tensor = torch.tensor([0.707, 0.0, 0.0, 0.707], device=self.device)  # sugar_box orientation (wxyz)

        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.clipped_actions = torch.zeros_like(self.actions)
        self.action_history_buf = torch.zeros(self.num_envs, 3, self.cfg.num_actions, device=self.device)
        self.command_history_buf = torch.zeros(self.num_envs, 3, self.cfg.num_actions, device=self.device)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.low_level_actions = torch.zeros(self.num_envs, 18, device=self.device)
        self.last_low_actions = torch.zeros(self.num_envs, 18, device=self.device)
        self.low_obs_buf = torch.zeros(self.num_envs, self.cfg.low_level_obs_space, device=self.device)
        self.low_obs_history_buf = torch.zeros(self.num_envs, 10, self.cfg.num_proprio, device=self.device)

        self.torques = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.p_gains = torch.zeros(self.robot.num_joints - self.num_gripper_joints, device=self.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        for i, name in enumerate(self.robot.joint_names[: -self.num_gripper_joints]):
            if "z1" in name:
                self.p_gains[i] = 5.0
                self.d_gains[i] = 0.5
            else:
                self.p_gains[i] = 80.0
                self.d_gains[i] = 2.0
        _scale_by_name = {
            "z1_waist": 2.1, "z1_shoulder": 0.6, "z1_elbow": 0.6,
            "z1_wrist_angle": 0.0, "z1_forearm_roll": 0.0, "z1_wrist_rotate": 0.0,
        }
        _leg_scale = {"hip_joint": 0.4, "thigh_joint": 0.45, "calf_joint": 0.45}
        self.action_scale = torch.tensor(
            [
                _scale_by_name.get(n, _leg_scale["thigh_joint" if "thigh" in n else ("calf_joint" if "calf" in n else "hip_joint")])
                for n in self.robot.joint_names[: -self.num_gripper_joints]
            ],
            device=self.device,
        )
        self.torque_limits = torch.tensor(
            [JOINT_EFFORT_LIMITS[n] for n in self.robot.joint_names], device=self.device
        )
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.num_gripper_joints, device=self.device)
        self.motor_strength = torch.ones(self.num_envs, 18, device=self.device)
        # default_joint_pos 是 (num_instances, num_joints)；[0] 去批（同 low_level_env.py:537）
        self.default_dof_pos = self.robot.data.default_joint_pos[0]
        self.default_dof_pos_wo_gripper = self.default_dof_pos[: -self.num_gripper_joints]
        self.last_dof_vel = torch.zeros_like(self.robot.data.joint_vel)

        self.gripper_dof_pos = torch.zeros(self.num_envs, self.num_gripper_joints, device=self.device)
        # joint_pos_limits 是 (num_instances, num_joints, 2)；[0] 取第一实例的 (num_joints, 2)
        jpl = self.robot.data.joint_pos_limits[0]
        self.dof_limits_lower = jpl[:, 0]
        self.dof_limits_upper = jpl[:, 1]

        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_orn_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_ee_goal_cart = torch.tensor([0.46, 0.0, 0.55], device=self.device).repeat(self.num_envs, 1)
        self.curr_ee_goal_cart[:] = self.init_ee_goal_cart[:]
        self.ee_goal_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_goal_cart_world = torch.zeros(self.num_envs, 3, device=self.device)

        # derived state (refreshed by ``_update_derived_state``; initialized so
        # ``_reset_idx`` / ``_pre_physics_step`` never touch undefined attributes)
        self.base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.base_yaw_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.arm_base = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_orn = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.last_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.base_obj_dis = torch.zeros(self.num_envs, device=self.device)
        self.closest_dist = -torch.ones(self.num_envs, device=self.device)
        self.curr_dist = torch.zeros(self.num_envs, device=self.device)
        self.curr_height = torch.zeros(self.num_envs, device=self.device)
        self.highest_object = -torch.ones(self.num_envs, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.lifted_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reach_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.pick_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # feet (canonical [FL, FR, RL, RR] order)
        self.foot_contact_forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.foot_contacts_from_sensor = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.is_walking = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.gait_wait_timer = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        # features
        self.feature_obs = torch.zeros(self.num_envs, 1024, device=self.device)
        feature_path = (
            "/home/haotian/workspace/visual_wholebody_isaaclab/source/visual_wholebody_isaaclab/visual_wholebody_isaaclab/assets/data/obj_set/sugar_box/features.npy"
        )
        feature = torch.from_numpy(np.load(feature_path, allow_pickle=True)).float().to(self.device)
        self.feature_obs[:] = feature[0]

        self.object_state = torch.zeros(self.num_envs, 13, device=self.device)
        self.object_root_state = torch.zeros(self.num_envs, 13, device=self.device)
        self.table_state = torch.zeros(self.num_envs, 13, device=self.device)

        self.eval_mode = False

    def _update_derived_state(self):
        self.base_quat = self.robot.data.root_quat_w
        self.arm_base = quat_apply(self.base_quat, self.arm_base_offset) + self.robot.data.root_pos_w
        _, _, yaw = euler_xyz_from_quat(self.base_quat)
        self.base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        self.ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id]
        self.ee_orn = self.robot.data.body_quat_w[:, self.ee_body_id]
        self.ee_goal_cart_world = quat_apply(self.base_quat, self.curr_ee_goal_cart) + self.arm_base
        self.ee_goal_orn_quat = quat_mul(
            self.base_quat,
            quat_from_euler_xyz(
                self.curr_ee_goal_orn_rpy[:, 0], self.curr_ee_goal_orn_rpy[:, 1], self.curr_ee_goal_orn_rpy[:, 2]
            ),
        )

        # object / table states
        self.object_state = self.object.data.root_state_w
        self.object_root_state = self.object.data.root_state_w
        self.table_height = torch.full_like(self.object_state[:, 2], self.table_z_center + self.table_half_height)

        # base-object distance
        base_obj_dis = self.object_state[:, :2] - self.arm_base[:, :2]
        self.base_obj_dis = torch.norm(base_obj_dis, dim=-1)

        # ee-object distance
        self.curr_dist[:] = torch.norm(self.ee_pos - self.object_state[:, :3], dim=-1)
        self.closest_dist = torch.where(self.closest_dist < 0, self.curr_dist, self.closest_dist)
        self.curr_height[:] = self.object_state[:, 2] - self.table_height - self.obj_height
        self.highest_object = torch.where(self.highest_object < 0, self.curr_height, self.highest_object)

        # foot contacts
        self.contact_forces = self.contact_sensor.data.net_forces_w
        self.foot_contact_forces = self.contact_forces[:, self.foot_contact_ids][:, self.feet_contact_perm]
        self.foot_contacts_from_sensor = self.foot_contact_forces.norm(dim=-1) > 2.0

    def _set_gripper(self):
        u_gripper = self.actions[:, 6].unsqueeze(-1)
        self.gripper_dof_pos[:] = torch.where(
            u_gripper >= 0, self.dof_limits_lower[-1].item(), self.dof_limits_upper[-1].item()
        )

    def _clip_commands(self):
        if not self.cfg.commands_curriculum:
            self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.4, 0.4)
            return
        if self.common_step_counter > 45000:
            self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.22, 0.22)
        elif self.common_step_counter > 30000:
            self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.3, 0.3)
        elif self.common_step_counter > 15000:
            self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.38, 0.38)
        else:
            self.commands[:, 0] = torch.clip(self.actions[:, 7], -0.45, 0.45)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.motor_strength * self.action_scale
        default_torques = (
            self.p_gains
            * (actions_scaled + self.default_dof_pos_wo_gripper - self.robot.data.joint_pos[:, :-self.num_gripper_joints])
            - self.d_gains * self.robot.data.joint_vel[:, :-self.num_gripper_joints]
        )
        default_torques[:, self.arm_joint_ids] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _control_ik(self, dpose):
        jacobians = self.robot.root_physx_view.get_jacobians()
        ee_jacobian = jacobians[:, self.ee_body_id, :6, [i + 6 for i in self.arm_joint_ids]]
        j_eef_T = torch.transpose(ee_jacobian, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05**2)
        A = torch.bmm(ee_jacobian, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))
        return u.squeeze(-1)

    def get_ee_goal_spherical_center(self):
        center = torch.cat(
            [self.robot.data.root_pos_w[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1
        )
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center

    def get_body_orientation(self, return_yaw=False):
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        body_angles = torch.stack([roll, pitch, yaw], dim=-1)
        if not return_yaw:
            return body_angles[:, :-1]
        return body_angles

    def _step_contact_targets(self):
        if not self.cfg.observe_gait_commands:
            return
        frequencies = 2
        phases = 0.5
        offsets = 0
        bounds = 0
        durations = 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.low_level_dt * frequencies, 1.0)
        is_walking = self.get_walking_cmd_mask()
        suddenstop_indices = (self.gait_wait_timer > 0) | ((~is_walking) & (self.is_walking))
        self.gait_indices[suddenstop_indices] += 1
        overdue_indices = self.gait_wait_timer >= 35
        self.gait_indices[overdue_indices] = 0.0
        self.gait_wait_timer[overdue_indices] = 0
        self.is_walking = is_walking
        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

    def get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > LIN_VEL_X_CLIP
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > ANG_VEL_PITCH_CLIP
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > ANG_VEL_YAW_CLIP
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask
