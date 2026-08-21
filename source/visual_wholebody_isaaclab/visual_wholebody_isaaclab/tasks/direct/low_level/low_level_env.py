# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DirectRL environment for the B1+Z1 low-level whole-body controller.

Direct port of ``ManipLoco`` from the original ``visual_wholebody`` Isaac Gym
project.

Control architecture (must be preserved):
    * The RL policy outputs 18 actions; columns ``12:`` are forced to zero.
    * The 12 leg joints are driven by the explicit PD controller
      ``_compute_original_leg_torques`` written with ``set_joint_effort_target``.
    * The 6 Z1 arm joints are driven by damped-least-squares differential IK,
      written with ``set_joint_position_target``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_rotate_inverse

from visual_wholebody_isaaclab.assets.b1_z1_cfg import (
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    FOOT_BODY_NAME,
    GRIPPER_JOINT_NAME,
    JOINT_EFFORT_LIMITS,
    LEG_JOINT_NAMES,
    PENALIZED_CONTACT_NAMES,
    TRAIN_JOINT_NAMES,
)
from visual_wholebody_isaaclab.utils.math_utils import (
    cart2sphere,
    orientation_error,
    sphere2cart,
    torch_rand_float,
)

from .low_level_env_cfg import VisualWholeBodyLowLevelEnvCfg
from .rewards import ManipLocoRewards


class VisualWholeBodyLowLevelEnv(DirectRLEnv):
    """Low-level B1+Z1 whole-body control environment."""

    cfg: VisualWholeBodyLowLevelEnvCfg

    def __init__(self, cfg: VisualWholeBodyLowLevelEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # resolve joints / bodies by name (never assume array order)
        self.leg_joint_ids, self.leg_joint_names = self.robot.find_joints(LEG_JOINT_NAMES)
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(ARM_JOINT_NAMES)
        self.gripper_joint_ids, _ = self.robot.find_joints(GRIPPER_JOINT_NAME)
        self.ee_body_id = self.robot.find_bodies(EE_BODY_NAME)[0][0]
        self.wrist_body_id = self.robot.find_bodies("link06")[0][0]
        # NOTE: articulation bodies and contact-sensor bodies live in two *different*
        # index spaces (``robot.data.*`` / ``robot.find_bodies`` vs
        # ``contact_sensor.data.net_forces_w`` / ``contact_sensor.find_bodies``).
        # They are NOT guaranteed to share the same ordering, so the two sets of
        # indices must never be mixed.
        self.foot_body_ids, self.foot_body_names = self.robot.find_bodies(f".*{FOOT_BODY_NAME}.*")
        self.foot_contact_ids, self.foot_contact_names = self.contact_sensor.find_bodies(f".*{FOOT_BODY_NAME}.*")
        # penalized bodies are only read from the contact sensor (``net_forces_w``)
        self.penalized_contact_body_ids = []
        for name in PENALIZED_CONTACT_NAMES:
            ids, _ = self.contact_sensor.find_bodies(f".*{name}.*")
            if len(ids) == 0:
                raise ValueError(f"No contact-sensor body found with name containing {name}")
            self.penalized_contact_body_ids.extend(ids)

        # hip joints (for hip_pos reward). These are indices into the *full* joint
        # tensor (robot/USD order), unlike the original which used URDF-order indices.
        self.hip_indices = torch.tensor(
            [self.robot.joint_names.index(n) for n in ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]],
            dtype=torch.long,
            device=self.device,
        )
        # robot-order indices of the 12 leg / 6 arm joints (for reward slicing)
        self.leg_indices = torch.tensor(self.leg_joint_ids, dtype=torch.long, device=self.device)
        self.arm_indices = torch.tensor(self.arm_joint_ids, dtype=torch.long, device=self.device)

        # Training-order permutation.
        # Isaac Lab orders DOFs by USD-tree depth (legs/arm interleaved), which differs
        # from the URDF file order. The original training observation orders the 18
        # non-gripper joints as legs-by-leg [FL, FR, RL, RR] then arm [z1_waist..z1_wrist_rotate].
        self.train_joint_names = list(TRAIN_JOINT_NAMES)
        # perm[i] = index in robot.joint_names of train_joint_names[i]
        self.perm = [self.robot.joint_names.index(n) for n in self.train_joint_names]
        self.perm = torch.tensor(self.perm, dtype=torch.long, device=self.device)
        # inverse: il_idx -> train_idx
        self.perm_inv = torch.zeros(len(self.robot.joint_names), dtype=torch.long, device=self.device)
        self.perm_inv[self.perm] = torch.arange(len(self.train_joint_names), device=self.device)

        # Feet are stored in a canonical [FL, FR, RL, RR] order everywhere downstream
        # (rewards AND observation), so positions/velocities and contact forces align
        # foot-by-foot. Build a reordering for *both* index spaces.
        foot_order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.feet_perm = torch.tensor(
            [self.foot_body_names.index(n) for n in foot_order], dtype=torch.long, device=self.device
        )
        self.feet_contact_perm = torch.tensor(
            [self.foot_contact_names.index(n) for n in foot_order], dtype=torch.long, device=self.device
        )

        self.num_gripper_joints = self.cfg.num_gripper_joints
        self.num_leg_joints = len(self.leg_joint_ids)
        self.num_arm_joints = len(self.arm_joint_ids)
        self.dt = self.cfg.decimation * self.cfg.sim.dt
        # NOTE: ``max_episode_length_s`` is a read-only property provided by DirectRLEnv
        # (it just returns ``cfg.episode_length_s``), used to normalize the per-reward
        # episode logs in ``_reset_idx`` (as in the original ``reset_idx``).
        # NOTE: max_episode_length is a read-only property provided by DirectRLEnv
        self.num_actions = self.cfg.action_space
        self.num_torques = self.cfg.num_torques
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_observations = self.cfg.normalization.clip_observations
        self.push_interval = int(math.ceil(self.cfg.domain_rand.push_interval_s / self.dt))
        self.action_delay = self.cfg.action_delay

        # buffers
        self._init_buffers()

        # compatibility aliases used by the reward container (ports of the original)
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel
        self.base_pos = self.robot.data.root_pos_w
        self.penalized_contact_indices = self.penalized_contact_body_ids

        # reward container (two channels: leg reward + arm reward, as in the original)
        self.reward_container = ManipLocoRewards(self)
        self.reward_scales = {
            "feet_air_time": self.cfg.rewards.feet_air_time,
            "feet_height": self.cfg.rewards.feet_height,
            "tracking_lin_vel_max": self.cfg.rewards.tracking_lin_vel_max,
            "tracking_ang_vel": self.cfg.rewards.tracking_ang_vel,
            "delta_torques": self.cfg.rewards.delta_torques,
            "torques": self.cfg.rewards.torques,
            "stand_still": self.cfg.rewards.stand_still,
            "walking_dof": self.cfg.rewards.walking_dof,
            "alive": self.cfg.rewards.alive,
            "lin_vel_z": self.cfg.rewards.lin_vel_z,
            "roll": self.cfg.rewards.roll,
            "ang_vel_xy": self.cfg.rewards.ang_vel_xy,
            "dof_acc": self.cfg.rewards.dof_acc,
            "collision": self.cfg.rewards.collision,
            "action_rate": self.cfg.rewards.action_rate,
            "dof_pos_limits": self.cfg.rewards.dof_pos_limits,
            "hip_pos": self.cfg.rewards.hip_pos,
            "work": self.cfg.rewards.work,
            "feet_jerk": self.cfg.rewards.feet_jerk,
            "feet_drag": self.cfg.rewards.feet_drag,
            "feet_contact_forces": self.cfg.rewards.feet_contact_forces,
            "base_height": self.cfg.rewards.base_height,
        }
        # arm reward channel (port of ``B1Z1RoughCfg.rewards.arm_scales``)
        self.arm_reward_scales = {
            "tracking_ee_world": self.cfg.rewards.tracking_ee_world,
            "tracking_ee_sphere": self.cfg.rewards.tracking_ee_sphere,
            "tracking_ee_orn": self.cfg.rewards.tracking_ee_orn,
        }
        self.reward_scales = {k: v for k, v in self.reward_scales.items() if v is not None and v != 0.0}
        self.arm_reward_scales = {k: v for k, v in self.arm_reward_scales.items() if v is not None and v != 0.0}
        self.reward_names = list(self.reward_scales.keys())
        self.reward_functions = [getattr(self.reward_container, f"_reward_{n}") for n in self.reward_names]
        self.arm_reward_names = list(self.arm_reward_scales.keys())
        self.arm_reward_functions = [getattr(self.reward_container, f"_reward_{n}") for n in self.arm_reward_names]

        # episode sums (for logging via extras)
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.reward_scales
        }
        self.episode_sums.update(
            {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.arm_reward_scales}
        )
        # per-reward metric sums (port of ``episode_metric_sums`` in the original reset_idx)
        self.episode_metric_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for name in self.reward_scales
        }
        self.episode_metric_sums.update(
            {
                name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for name in self.arm_reward_scales
            }
        )


        # apply domain randomization
        self._apply_domain_rand()

    # ------------------------------------------------------------------ setup

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # EE-goal debug markers (port of the original ``_draw_ee_goal_curr``): shown when a
        # viewer is open (rendered training / play) or when debug_viz / record_video is set.
        self._ee_goal_markers = None
        if self.cfg.debug_viz or self.cfg.record_video or self.sim.has_gui():
            self._init_ee_goal_markers()

    def _init_ee_goal_markers(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._ee_goal_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/EEGoalMarkers",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(1.0, 1.0, 0.0)),
                    ),
                    "ee": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(0.0, 0.0, 1.0)),
                    ),
                    "center": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(emissive_color=(0.0, 1.0, 1.0)),
                    ),
                },
            )
        )

    def _update_ee_goal_markers(self):
        if self._ee_goal_markers is None:
            return
        n = self.num_envs
        # yellow = current EE goal, blue = current EE, cyan = spherical center (arm base)
        translations = torch.stack(
            [self.curr_ee_goal_cart_world, self.ee_pos, self._get_ee_goal_spherical_center()], dim=1
        ).reshape(-1, 3)
        marker_indices = torch.tensor([0, 1, 2] * n, dtype=torch.long, device=self.device)
        self._ee_goal_markers.visualize(marker_indices=marker_indices, translations=translations)

    # ---------------------------------------------------------- step overrides

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # policy actions are in training order (legs [FL,FR,RL,RR] then arm)
        actions = actions.clone()
        actions[:, 12:] = 0.0
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)

        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
        if self.common_step_counter < self.cfg.action_delay_start_step:
            delayed_actions = self.action_history_buf[:, -1]
        else:
            delayed_actions = self.action_history_buf[:, -2]

        delayed_actions = delayed_actions.clone()
        delayed_actions[:, 12:] = 0.0
        self.actions_train = delayed_actions  # training order (stored for the observation history)

        # scatter training-order actions into the robot joint order used for torques
        il_actions = torch.zeros_like(delayed_actions)
        il_actions[:, self.perm] = delayed_actions
        self.actions = il_actions

        # compute arm IK targets once per RL step (matches the original).
        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        ee_orn = self.robot.data.body_quat_w[:, self.ee_body_id]
        ee_orn = ee_orn / torch.norm(ee_orn, dim=-1).unsqueeze(-1)
        drot = orientation_error(self.ee_goal_orn_quat, ee_orn)
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        self.arm_pos_targets = self._control_ik(dpose) + self.robot.data.joint_pos[:, self.arm_joint_ids]

    def _apply_action(self) -> None:
        # legs: explicit PD torques
        self.torques = self._compute_torques(self.actions)
        self.robot.set_joint_effort_target(self.torques, joint_ids=None)
        # arm: position targets from differential IK
        self.robot.set_joint_position_target(self.arm_pos_targets, joint_ids=self.arm_joint_ids)
        # gripper: hold at zero position target (as in the original)
        self.robot.set_joint_position_target(
            torch.zeros(self.num_envs, self.num_gripper_joints, device=self.device), joint_ids=self.gripper_joint_ids
        )

    def _get_observations(self) -> dict:
        if self.cfg.stand_by:
            # port of the original ``compute_observations``: in stand-by mode the
            # velocity/angular commands are held at zero every step.
            self.commands[:] = 0.0

        # compute base-frame quantities
        self.base_quat = self.robot.data.root_quat_w
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot.data.root_lin_vel_w)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot.data.root_ang_vel_w)
        _, _, yaw = euler_xyz_from_quat(self.base_quat)
        self.base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        # ee goal local cart
        arm_base_pos = self.robot.data.root_pos_w + quat_apply(self.base_yaw_quat, self.arm_base_offset_tensor)
        ee_goal_local_cart = quat_rotate_inverse(
            self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos
        )

        obs_buf = torch.cat(
            (
                self._get_body_orientation(),  # dim 2
                self.base_ang_vel * self.obs_scales_ang_vel,  # dim 3
                (self.robot.data.joint_pos - self.default_dof_pos)[:, self.perm] * self.obs_scales_dof_pos,  # dim 18
                self.robot.data.joint_vel[:, self.perm] * self.obs_scales_dof_vel,  # dim 18
                self.action_history_buf[:, -1, :12],  # dim 12 (training-order legs)
                self.foot_contacts_from_sensor,  # dim 4, already canonical [FL, FR, RL, RR]
                self.commands[:, :3] * self.commands_scale,  # dim 3
                ee_goal_local_cart,  # dim 3
                0.0 * self.curr_ee_goal_sphere,  # dim 3 orientation (zeros)
            ),
            dim=-1,
        )
        if self.cfg.observe_gait_commands:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)

        if self.cfg.domain_rand.observe_priv:
            priv_buf = torch.cat(
                (
                    self.mass_params_tensor,
                    self.friction_coeffs_tensor,
                    self.motor_strength[:, self.leg_joint_ids] - 1,
                ),
                dim=-1,
            )
            self.obs_buf = torch.cat([obs_buf, priv_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self.obs_buf = torch.clip(self.obs_buf, -self.clip_observations, self.clip_observations)

        # update last_* buffers (end of step, matches original ordering)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.robot.data.joint_vel[:]
        self.last_torques[:] = self.torques[:]

        return {"policy": self.obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        # leg reward channel
        self.rew_buf[:] = 0.0
        for i, name in enumerate(self.reward_names):
            rew, metric = self.reward_functions[i]()
            rew = rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        self.rew_buf /= 100.0

        # arm reward channel
        self.arm_rew_buf[:] = 0.0
        for i, name in enumerate(self.arm_reward_names):
            rew, metric = self.arm_reward_functions[i]()
            rew = rew * self.arm_reward_scales[name]
            self.arm_rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric
        if self.cfg.rewards.only_positive_rewards:
            self.arm_rew_buf[:] = torch.clip(self.arm_rew_buf[:], min=0.0)
        self.arm_rew_buf /= 100.0

        # DirectRLEnv requires a single scalar reward; the v1 VecEnv wrapper reads
        # the two channels separately (rew_buf for legs, arm_rew_buf for the arm).
        return self.rew_buf + self.arm_rew_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # post-physics callback: resample commands, gait, push robots, update ee goal
        self._post_physics_step_callback()
        self._update_curr_ee_goal()

        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        z = self.robot.data.root_pos_w[:, 2]
        r_term = torch.abs(roll) > self.cfg.termination.r_threshold
        p_term = torch.abs(pitch) > self.cfg.termination.p_threshold
        z_term = z < self.cfg.termination.z_threshold
        terminated = r_term | p_term | z_term

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset root states
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :2] += torch_rand_float(
            -self.cfg.init_state.origin_perturb_range,
            self.cfg.init_state.origin_perturb_range,
            (len(env_ids), 2),
            self.device,
        )
        rand_yaw = self.cfg.init_state.rand_yaw_range * torch_rand_float(-1.0, 1.0, (len(env_ids), 1), self.device).squeeze(1)
        yaw_quat = quat_from_euler_xyz(0.0 * rand_yaw, 0.0 * rand_yaw, rand_yaw)
        default_root_state[:, 3:7] = yaw_quat
        default_root_state[:, 7:13] = torch_rand_float(
            -self.cfg.init_state.init_vel_perturb_range,
            self.cfg.init_state.init_vel_perturb_range,
            (len(env_ids), 6),
            self.device,
        )

        # reset joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids] * torch_rand_float(
            0.8, 1.2, (len(env_ids), self.robot.num_joints), self.device
        )
        joint_vel = torch.zeros_like(joint_pos)

        # write into data buffers so observations reflect the reset state
        self.robot.data.root_pos_w[env_ids] = default_root_state[:, :3]
        self.robot.data.root_quat_w[env_ids] = default_root_state[:, 3:7]
        self.robot.data.root_lin_vel_w[env_ids] = default_root_state[:, 7:10]
        self.robot.data.root_ang_vel_w[env_ids] = default_root_state[:, 10:13]
        self.robot.data.joint_pos[env_ids] = joint_pos
        self.robot.data.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # refresh buffers so body states reflect the reset configuration
        self.scene.update(dt=self.physics_dt)

        # resample commands (only on time-outs, as in the original) and ee goal
        command_env_ids = env_ids[self.reset_time_outs[env_ids]]
        self._resample_commands(command_env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        # reset buffers
        self.last_torques[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.obs_history_buf[env_ids, :, :] = 0.0
        self.action_history_buf[env_ids, :, :] = 0.0
        self.goal_timer[env_ids] = 0.0

        # fill extras (per-reward episode logging, port of the original ``reset_idx``:
        # rew_* / metric_* normalized by the episode length in seconds)
        if self.episode_sums:
            self.extras["episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
                )
                self.episode_sums[key][env_ids] = 0.0
            for key in self.episode_metric_sums.keys():
                self.extras["episode"]["metric_" + key] = (
                    torch.mean(self.episode_metric_sums[key][env_ids]) / self.max_episode_length_s
                )
                self.episode_metric_sums[key][env_ids] = 0.0

    # ------------------------------------------------------------- initialization

    def _init_buffers(self):
        self.obs_scales_ang_vel = self.cfg.normalization.ang_vel
        self.obs_scales_dof_pos = self.cfg.normalization.dof_pos
        self.obs_scales_dof_vel = self.cfg.normalization.dof_vel
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.lin_vel,
                self.cfg.normalization.lin_vel,
                self.cfg.normalization.ang_vel,
            ],
            device=self.device,
        )[: self.cfg.commands.num_commands]

        self.add_noise = self.cfg.noise.add_noise

        # PD gains (legs: 80/2.0, arm: 5/0.5) in robot joint order
        self.p_gains = torch.zeros(self.robot.num_joints - self.num_gripper_joints, device=self.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        joint_names = self.robot.joint_names
        for i, name in enumerate(joint_names[: -self.num_gripper_joints]):
            if "z1" in name:
                self.p_gains[i] = self.cfg.control.stiffness_z1
                self.d_gains[i] = self.cfg.control.damping_z1
            else:
                self.p_gains[i] = self.cfg.control.stiffness_joint
                self.d_gains[i] = self.cfg.control.damping_joint

        # action scale in robot joint order (all legs share the same [hip,thigh,calf] scale)
        _scale_by_name = {
            "z1_waist": 2.1, "z1_shoulder": 0.6, "z1_elbow": 0.6,
            "z1_wrist_angle": 0.0, "z1_forearm_roll": 0.0, "z1_wrist_rotate": 0.0,
        }
        _leg_scale = {"hip_joint": 0.4, "thigh_joint": 0.45, "calf_joint": 0.45}
        self.action_scale = torch.tensor(
            [
                _scale_by_name.get(n, _leg_scale[("thigh_joint" if "thigh" in n else ("calf_joint" if "calf" in n else "hip_joint"))])
                for n in joint_names[: -self.num_gripper_joints]
            ],
            device=self.device,
        )
        self.torque_limits = torch.tensor(
            [JOINT_EFFORT_LIMITS[n] for n in self.robot.joint_names], device=self.device
        )
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.num_gripper_joints, device=self.device)

        # soft joint limits (from URDF joint_pos_limits, unbatched -> (num_joints, 2))
        jpl = self.robot.data.joint_pos_limits[0]
        m = (jpl[:, 0] + jpl[:, 1]) / 2
        r = jpl[:, 1] - jpl[:, 0]
        self.dof_pos_limits = torch.stack(
            [m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit, m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit],
            dim=1,
        )

        # default joint pos is batched (num_envs, num_joints); take the per-joint vector
        self.default_dof_pos = self.robot.data.default_joint_pos[0]
        self.default_dof_pos_wo_gripper = self.default_dof_pos[: -self.num_gripper_joints]

        # action / observation history
        self.obs_history_buf = torch.zeros(
            self.num_envs, self.cfg.history_len, self.cfg.num_proprio, device=self.device
        )
        self.action_history_buf = torch.zeros(
            self.num_envs, self.action_delay + 2, self.num_actions, device=self.device
        )

        # commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.command_ranges = {
            "lin_vel_x": self.cfg.commands.lin_vel_x,
            "ang_vel_yaw": self.cfg.commands.ang_vel_yaw,
        }

        # arm / ee
        self.arm_base_offset = self.cfg.arm_base_offset
        self.arm_base_offset_tensor = torch.tensor(self.cfg.arm_base_offset, device=self.device).repeat(self.num_envs, 1)
        self.ee_goal_center_offset = torch.tensor(
            [
                self.cfg.goal_ee.x_offset,
                self.cfg.goal_ee.y_offset,
                self.cfg.goal_ee.z_invariant_offset,
            ],
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.goal_ee_ranges = {
            "pos_l": self.cfg.goal_ee.pos_l,
            "pos_p": self.cfg.goal_ee.pos_p,
            "pos_y": self.cfg.goal_ee.pos_y,
            "delta_orn_r": self.cfg.goal_ee.delta_orn_r,
            "delta_orn_p": self.cfg.goal_ee.delta_orn_p,
            "delta_orn_y": self.cfg.goal_ee.delta_orn_y,
        }
        self.traj_timesteps = (
            torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), self.device)
            .squeeze(1)
            / self.dt
        )
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(
            self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), self.device
        ).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler[:, 0] = np.pi / 2
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            self.ee_goal_orn_euler[:, 0], self.ee_goal_orn_euler[:, 1], self.ee_goal_orn_euler[:, 2]
        )
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_start_ee_sphere = torch.tensor(self.cfg.goal_ee.init_pos_start, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.cfg.goal_ee.init_pos_end, device=self.device).unsqueeze(0)
        self.sphere_error_scale = torch.tensor(self.cfg.goal_ee.sphere_error_scale, device=self.device)
        self.orn_error_scale = torch.tensor(self.cfg.goal_ee.orn_error_scale, device=self.device)
        self.collision_upper_limits = torch.tensor(
            self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float
        )
        self.collision_lower_limits = torch.tensor(
            self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float
        )
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[
            None, None, :
        ]

        # torques / actions
        self.torques = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.robot.data.joint_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.arm_rew_buf = torch.zeros(self.num_envs, device=self.device)

        # feet (canonical [FL, FR, RL, RR] order)
        self.foot_positions = self.robot.data.body_pos_w[:, self.foot_body_ids][:, self.feet_perm]
        self.foot_velocities = self.robot.data.body_lin_vel_w[:, self.foot_body_ids][:, self.feet_perm]
        self.foot_contact_forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.foot_contacts_from_sensor = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        # ee state (refreshed every step)
        self.ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id]
        self.ee_orn = self.robot.data.body_quat_w[:, self.ee_body_id]

        # domain randomization tensors
        self.mass_params_tensor = torch.zeros(self.num_envs, 5, device=self.device)
        self.friction_coeffs_tensor = torch.ones(self.num_envs, 1, device=self.device)
        self.motor_strength = torch.ones(self.num_envs, self.num_torques, device=self.device)

        # gait (only used when observe_gait_commands is True)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)

        # noise scale
        self.noise_scale_vec = self._get_noise_scale_vec()

        # gravity vector
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

    def _apply_domain_rand(self):
        # base / gripper mass and base com
        # NOTE: the PhysX articulation view expects CPU tensors + int32 indices.
        indices = torch.arange(self.num_envs, dtype=torch.int32)
        if self.cfg.domain_rand.randomize_base_mass or self.cfg.domain_rand.randomize_base_com:
            trunk_id = self.robot.find_bodies("trunk")[0][0]
            masses = self.robot.root_physx_view.get_masses()
            coms = self.robot.root_physx_view.get_coms()
            rand_mass = np.zeros(self.num_envs)
            rand_com = np.zeros((self.num_envs, 3))
            if self.cfg.domain_rand.randomize_base_mass:
                rand_mass = np.random.uniform(*self.cfg.domain_rand.added_mass_range, size=(self.num_envs,))
            if self.cfg.domain_rand.randomize_base_com:
                rand_com[:, 0] = np.random.uniform(*self.cfg.domain_rand.added_com_range_x, size=(self.num_envs,))
                rand_com[:, 1] = np.random.uniform(*self.cfg.domain_rand.added_com_range_y, size=(self.num_envs,))
                rand_com[:, 2] = np.random.uniform(*self.cfg.domain_rand.added_com_range_z, size=(self.num_envs,))
            masses[:, trunk_id] += torch.from_numpy(rand_mass).float()
            coms[:, trunk_id, :3] += torch.from_numpy(rand_com).float()
            self.robot.root_physx_view.set_masses(masses, indices)
            self.robot.root_physx_view.set_coms(coms, indices)

        if self.cfg.domain_rand.randomize_gripper_mass:
            gripper_mass = np.random.uniform(*self.cfg.domain_rand.gripper_added_mass_range, size=(self.num_envs,))
            masses = self.robot.root_physx_view.get_masses()
            masses[:, self.ee_body_id] += torch.from_numpy(gripper_mass).float()
            self.robot.root_physx_view.set_masses(masses, indices)

        if self.cfg.domain_rand.randomize_motor:
            # build motor strength in robot joint order (legs vs arm use different ranges)
            leg_lo, leg_hi = self.cfg.domain_rand.leg_motor_strength_range
            arm_lo, arm_hi = self.cfg.domain_rand.arm_motor_strength_range
            strengths = []
            for name in self.robot.joint_names[: -self.num_gripper_joints]:
                lo, hi = (arm_lo, arm_hi) if "z1" in name else (leg_lo, leg_hi)
                strengths.append(torch_rand_float(lo, hi, (self.num_envs, 1), self.device))
            self.motor_strength = torch.cat(strengths, dim=1)
        self.mass_params_tensor = torch.cat(
            [
                torch.tensor(
                    np.random.uniform(0.0, 15.0, size=(self.num_envs, 1)), dtype=torch.float, device=self.device
                )
                if self.cfg.domain_rand.randomize_base_mass
                else torch.zeros(self.num_envs, 1, device=self.device),
                torch.tensor(
                    np.random.uniform(-0.15, 0.15, size=(self.num_envs, 3)), dtype=torch.float, device=self.device
                )
                if self.cfg.domain_rand.randomize_base_com
                else torch.zeros(self.num_envs, 3, device=self.device),
                torch.tensor(
                    np.random.uniform(0.0, 0.1, size=(self.num_envs, 1)), dtype=torch.float, device=self.device
                )
                if self.cfg.domain_rand.randomize_gripper_mass
                else torch.zeros(self.num_envs, 1, device=self.device),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------- observation helpers

    def _get_body_orientation(self, return_yaw=False):
        roll, pitch, yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        body_angles = torch.stack([roll, pitch, yaw], dim=-1)
        if not return_yaw:
            return body_angles[:, :-1]
        return body_angles

    def _get_walking_cmd_mask(self, env_ids=None, return_all=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        walking_mask0 = torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip
        walking_mask1 = torch.abs(self.commands[env_ids, 1]) > self.cfg.commands.lin_vel_x_clip
        walking_mask2 = torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2
        if return_all:
            return walking_mask0, walking_mask1, walking_mask2, walking_mask
        return walking_mask

    def _get_noise_scale_vec(self):
        noise_scales = self.cfg.noise
        noise_level = noise_scales.noise_level
        noise_vec = torch.zeros(self.cfg.observation_space, device=self.device)
        idx = 0
        noise_vec[idx : idx + 2] = 0
        idx += 2
        noise_vec[idx : idx + 3] = noise_scales.ang_vel * noise_level * self.obs_scales_ang_vel
        idx += 3
        noise_vec[idx : idx + 12] = noise_scales.dof_pos * noise_level * self.obs_scales_dof_pos
        idx += 12
        noise_vec[idx : idx + 6] = 0
        idx += 6
        noise_vec[idx : idx + 12] = noise_scales.dof_vel * noise_level * self.obs_scales_dof_vel
        idx += 12
        noise_vec[idx : idx + 6] = 0
        idx += 6
        noise_vec[idx : idx + 12] = 0
        idx += 12
        noise_vec[idx : idx + 4] = 0
        idx += 4
        noise_vec[idx : idx + 3] = 0
        idx += 3
        noise_vec[idx : idx + 3] = 0
        idx += 3
        noise_vec[idx : idx + 3] = 0
        idx += 3
        return noise_vec

    # ------------------------------------------------------------- control

    def _compute_torques(self, actions):
        actions_scaled = actions * self.motor_strength * self.action_scale
        default_torques = (
            self.p_gains
            * (actions_scaled + self.default_dof_pos_wo_gripper - self.robot.data.joint_pos[:, :-self.num_gripper_joints])
            - self.d_gains * self.robot.data.joint_vel[:, :-self.num_gripper_joints]
        )
        # arm torques are zeroed (driven by IK position targets instead)
        default_torques[:, self.arm_joint_ids] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _control_ik(self, dpose):
        # damped least squares IK
        jacobians = self.robot.root_physx_view.get_jacobians()
        ee_jacobian = jacobians[:, self.ee_body_id, :6, [i + 6 for i in self.arm_joint_ids]]
        j_eef_T = torch.transpose(ee_jacobian, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05**2)
        A = torch.bmm(ee_jacobian, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))
        return u.squeeze(-1)

    # ------------------------------------------------------------- ee goal

    def _get_ee_goal_spherical_center(self):
        center = torch.cat(
            [self.robot.data.root_pos_w[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1
        )
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center

    def _update_curr_ee_goal(self):
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + ee_goal_cart_yaw_global

        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.cfg.goal_ee.arm_induced_pitch
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2,
            default_pitch + self.ee_goal_orn_delta_rpy[:, 1],
            self.ee_goal_orn_delta_rpy[:, 2] + default_yaw,
        )

        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        if len(resample_id) > 0 and self.cfg.stop_update_goal:
            self.commands[resample_id, 0] = 0
            self.commands[resample_id, 2] = 0
        self._resample_ee_goal(resample_id)
        self._update_ee_goal_markers()

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(
            self.goal_ee_ranges["pos_l"][0], self.goal_ee_ranges["pos_l"][1], (len(env_ids), 1), self.device
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(
            self.goal_ee_ranges["pos_p"][0], self.goal_ee_ranges["pos_p"][1], (len(env_ids), 1), self.device
        ).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(
            self.goal_ee_ranges["pos_y"][0], self.goal_ee_ranges["pos_y"][1], (len(env_ids), 1), self.device
        ).squeeze(1)

    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(
            self.goal_ee_ranges["delta_orn_r"][0], self.goal_ee_ranges["delta_orn_r"][1], (len(env_ids), 1), self.device
        )
        ee_goal_delta_orn_p = torch_rand_float(
            self.goal_ee_ranges["delta_orn_p"][0], self.goal_ee_ranges["delta_orn_p"][1], (len(env_ids), 1), self.device
        )
        ee_goal_delta_orn_y = torch_rand_float(
            self.goal_ee_ranges["delta_orn_y"][0], self.goal_ee_ranges["delta_orn_y"][1], (len(env_ids), 1), self.device
        )
        self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat(
            [ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1
        )

    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) == 0:
            return
        init_env_ids = env_ids.clone()
        if is_init:
            self.ee_goal_orn_delta_rpy[env_ids, :] = 0
            self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[:]
            self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere[:]
        else:
            self._resample_ee_goal_orn_once(env_ids)
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for _ in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self._collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
        self.ee_goal_cart[init_env_ids, :] = sphere2cart(self.ee_goal_sphere[init_env_ids, :])
        self.goal_timer[init_env_ids] = 0.0

    def _collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ..., None], self.collision_check_t
        ).squeeze(-1)
        ee_target_cart = sphere2cart(
            torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)
        ).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(
            torch.logical_and(
                torch.all(ee_target_cart < self.collision_upper_limits, dim=-1),
                torch.all(ee_target_cart > self.collision_lower_limits, dim=-1),
            ),
            dim=0,
        )
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask

    # ------------------------------------------------------------- commands / post-physics

    def _post_physics_step_callback(self):
        command_env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(command_env_ids)
        self._step_contact_targets()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.push_interval == 0):
            self._push_robots()

        # refresh ee / foot derived state used by rewards and observations
        self.base_quat = self.robot.data.root_quat_w
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot.data.root_lin_vel_w)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot.data.root_ang_vel_w)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        _, _, yaw = euler_xyz_from_quat(self.base_quat)
        self.base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        self.ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id]
        self.ee_orn = self.robot.data.body_quat_w[:, self.ee_body_id]
        self.foot_positions = self.robot.data.body_pos_w[:, self.foot_body_ids][:, self.feet_perm]
        self.foot_velocities = self.robot.data.body_lin_vel_w[:, self.foot_body_ids][:, self.feet_perm]

        self.contact_forces = self.contact_sensor.data.net_forces_w
        self.foot_contact_forces = self.contact_forces[:, self.foot_contact_ids][:, self.feet_contact_perm]
        self.foot_contacts_from_sensor = self.foot_contact_forces.norm(dim=-1) > 1.5

    def _resample_commands(self, env_ids):
        if self.cfg.teleop_mode or len(env_ids) == 0:
            return
        if self.common_step_counter < 5000 * 24:
            self.commands[env_ids, 0] = torch_rand_float(
                0, self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), self.device
            ).squeeze(1)
        else:
            self.commands[env_ids, 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0],
                self.command_ranges["lin_vel_x"][1],
                (len(env_ids), 1),
                self.device,
            ).squeeze(1)

        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), self.device
        ).squeeze(1)

        self.commands[env_ids, :] *= (
            torch.logical_or(
                torch.abs(self.commands[env_ids, 0]) > self.cfg.commands.lin_vel_x_clip,
                torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_yaw_clip,
            )
        ).unsqueeze(1)

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        lin = self.robot.data.root_lin_vel_w.clone()
        lin[:, :2] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), self.device)
        lin[:, :2] = torch.where(
            self.commands.sum(dim=1).unsqueeze(-1) == 0,
            lin[:, :2] * 2.5,
            lin[:, :2],
        )
        vel = torch.cat([lin, self.robot.data.root_ang_vel_w], dim=-1)
        self.robot.write_root_velocity_to_sim(vel)

    def _step_contact_targets(self):
        if not self.cfg.observe_gait_commands:
            return
        frequencies = self.cfg.frequencies
        phases = 0.5
        offsets = 0
        bounds = 0
        durations = 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        self.gait_indices[~self._get_walking_cmd_mask()] = 0

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]
        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (0.5 / (1 - durations))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
        smoothing_multiplier_FL = (
            smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1))
        )
        smoothing_multiplier_FR = (
            smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1))
        )
        smoothing_multiplier_RL = (
            smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1))
        )
        smoothing_multiplier_RR = (
            smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1))
        )
        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR


from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402
