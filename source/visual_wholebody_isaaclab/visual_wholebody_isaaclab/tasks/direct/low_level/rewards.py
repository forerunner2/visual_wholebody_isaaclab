# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the B1+Z1 low-level environment.

Direct port of ``ManipLoco_rewards`` from the original ``visual_wholebody``
project. Each function returns ``(reward, metric)``; the reward is multiplied by
its scale in the environment. Reward scales are NOT multiplied by ``dt`` (as in
the original ``ManipLoco``), and the final total reward is divided by 100.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_rotate_inverse

from visual_wholebody_isaaclab.utils.math_utils import cart2sphere, torch_wrap_to_pi_minuspi


class ManipLocoRewards:
    """Reward container for the low-level manip-loco environment."""

    def __init__(self, env):
        self.env = env

    # -------------Z1: Reward functions----------------

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - self.env.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error / self.env.cfg.rewards.tracking_ee_sigma * 2)
        return rew, ee_pos_error

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(
            self.env.base_yaw_quat, self.env.ee_pos - self.env.get_ee_goal_spherical_center()
        )
        ee_pos_error = torch.sum(
            torch.abs(cart2sphere(ee_pos_local) - self.env.curr_ee_goal_sphere) * self.env.sphere_error_scale, dim=1
        )
        return torch.exp(-ee_pos_error / self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_orn(self):
        roll, pitch, yaw = euler_xyz_from_quat(self.env.ee_orn)
        ee_orn_euler = torch.stack([roll, pitch, yaw], dim=-1)
        orn_err = torch.sum(
            torch.abs(torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler)) * self.env.orn_error_scale,
            dim=1,
        )
        return torch.exp(-orn_err / self.env.cfg.rewards.tracking_ee_sigma), orn_err

    def _reward_tracking_ee_orn_ry(self):
        roll, pitch, yaw = euler_xyz_from_quat(self.env.ee_orn)
        ee_orn_euler = torch.stack([roll, pitch, yaw], dim=-1)
        orn_err = torch.sum(
            torch.abs((torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler) * self.env.orn_error_scale)[
                :, [0, 2]
            ]),
            dim=1,
        )
        return torch.exp(-orn_err / self.env.cfg.rewards.tracking_ee_sigma), orn_err

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(
            torch.abs(
                self.env.torques[:, self.env.arm_indices]
                * self.env.dof_vel[:, self.env.arm_indices]
            ),
            dim=1,
        )
        return energy, energy

    # -------------B1: Reward functions----------------

    def _reward_tracking_lin_vel_max(self):
        rew = torch.where(
            self.env.commands[:, 0] > 0,
            torch.minimum(self.env.base_lin_vel[:, 0], self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5),
            torch.minimum(-self.env.base_lin_vel[:, 0], -self.env.commands[:, 0]) / (-self.env.commands[:, 0] + 1e-5),
        )
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < self.env.cfg.commands.lin_vel_x_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(self.env.base_lin_vel[:, 0]))[zero_cmd_indices]
        return rew, rew

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma), ang_vel_error

    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.env.torques - self.env.last_torques)[:, self.env.leg_indices], dim=1)
        return rew, rew

    def _reward_torques(self):
        torque = torch.sum(torch.square(self.env.torques), dim=1)
        return torque, torque

    def _reward_stand_still(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, self.env.leg_indices], dim=1)
        # NOTE: exponent hardened 0.05 -> 0.5 vs the original (fix for the "3-leg stance" reward
        # plateau: with 0.05 raising a whole leg costs ~4% of this term, which the `work` savings
        # outweighs). Only `stand_still` is hardened -- `walking_dof` keeps 0.05 so the gait is
        # not fought during locomotion.
        rew = torch.exp(-dof_error * 0.5)
        rew[self.env._get_walking_cmd_mask()] = 0.0
        return rew, rew

    def _reward_walking_dof(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, self.env.leg_indices], dim=1)
        rew = torch.exp(-dof_error * 0.05)
        rew[~self.env._get_walking_cmd_mask()] = 0.0
        return rew, rew

    def _reward_alive(self):
        return 1.0, 1.0

    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        return rew, rew

    def _reward_roll(self):
        roll = self.env._get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error, error

    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
        return rew, rew

    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, self.env.leg_indices] / self.env.dt), dim=1)
        return rew, rew

    def _reward_collision(self):
        rew = torch.sum(
            1.0 * (torch.norm(self.env.contact_forces[:, self.env.penalized_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )
        return rew, rew

    def _reward_action_rate(self):
        action_rate = torch.sum(torch.square(self.env.last_actions - self.env.actions)[:, self.env.leg_indices], dim=1)
        return action_rate, action_rate

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.0)
        rew = torch.sum(out_of_limits[:, self.env.leg_indices], dim=1)
        return rew, rew

    def _reward_hip_pos(self):
        rew = torch.sum(
            torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]),
            dim=1,
        )
        return rew, rew

    def _reward_work(self):
        work = self.env.torques * self.env.dof_vel
        abs_sum_work = torch.abs(torch.sum(work[:, self.env.leg_indices], dim=1))
        return abs_sum_work, abs_sum_work

    def _reward_feet_jerk(self):
        if not hasattr(self, "last_contact_forces"):
            result = torch.zeros(self.env.num_envs, device=self.env.device)
        else:
            result = torch.sum(torch.norm(self.env.foot_contact_forces - self.env.last_contact_forces, dim=-1), dim=-1)
        self.env.last_contact_forces = self.env.foot_contact_forces.clone()
        result[self.env.episode_length_buf < 50] = 0.0
        return result, result

    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.env.foot_velocities).sum(dim=-1)
        dragging_vel = self.env.foot_contacts_from_sensor * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew, rew

    def _reward_feet_contact_forces(self):
        reset_flag = (self.env.episode_length_buf > 2.0 / self.env.dt).type(torch.float)
        forces = torch.sum(
            (torch.norm(self.env.foot_contact_forces, dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0),
            dim=-1,
        )
        rew = reset_flag * forces
        return rew, rew

    def _reward_base_height(self):
        base_height = torch.mean(self.env.robot.data.root_pos_w[:, 2].unsqueeze(1), dim=1)
        return torch.abs(base_height - self.env.cfg.rewards.base_height_target), base_height

    def _reward_penalty_lin_vel_y(self):
        rew = torch.abs(self.env.base_lin_vel[:, 1])
        rot_indices = torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.ang_vel_yaw_clip
        rew[rot_indices] = 0.0
        return rew, rew

    def _reward_feet_air_time(self):
        first_contact = (self.env.feet_air_time > 0.0) * self.env.foot_contacts_from_sensor
        self.env.feet_air_time += self.env.dt

        if self.env.cfg.rewards.feet_aritime_allfeet:
            rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        else:
            rew_airTime = torch.sum((self.env.feet_air_time[:, :2] - 0.5) * first_contact[:, :2], dim=1)

        rew_airTime *= self.env._get_walking_cmd_mask()
        self.env.feet_air_time *= ~self.env.foot_contacts_from_sensor
        return rew_airTime, rew_airTime

    def _reward_feet_height(self):
        feet_height_tracking = self.env.cfg.rewards.feet_height_target
        if self.env.cfg.rewards.feet_height_allfeet:
            feet_height = self.env.foot_positions[:, :, 2]
        else:
            feet_height = self.env.foot_positions[:, :2, 2]

        rew = torch.clamp(torch.norm(feet_height, dim=-1) - feet_height_tracking, max=0.0)
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0.0
        return rew, rew

    def _reward_tracking_contacts_shaped_force(self):
        if not self.env.cfg.observe_gait_commands:
            return 0.0, 0.0
        # NOTE: ``contact_forces`` is in the contact-sensor index space, so it must be
        # sliced with the *sensor* foot ids and then permuted to the canonical
        # [FL, FR, RL, RR] order to line up per-foot with ``desired_contact_states``.
        # (``self.env.feet_indices`` does not exist here -- same index-space bug class
        # as the P0 contact-sensor fix in ``low_level_env.py``.)
        foot_forces = torch.norm(
            self.env.contact_forces[:, self.env.foot_contact_ids][:, self.env.feet_contact_perm], dim=-1
        )
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += -(1 - desired_contact[:, i]) * (
                1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma)
            )
        return reward / 4, reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.env.cfg.observe_gait_commands:
            return 0.0, 0.0
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += -(
                desired_contact[:, i]
                * (1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma))
            )
        return reward / 4, reward / 4

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error / self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error
