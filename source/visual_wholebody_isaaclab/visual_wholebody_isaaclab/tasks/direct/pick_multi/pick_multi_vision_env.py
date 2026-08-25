# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual (camera) student environment for the B1+Z1 pick-multi task.

Extends the state-teacher environment with two TiledCameras (front + wrist) and
replaces the observation with the image pipeline:

    forward_mask, wrist_mask, forward_masked_depth, wrist_masked_depth

with a history of ``camera_history_len`` (3), i.e. 12 channels in total.

NOTE: depth convention and semantic-id resolution MUST be verified at runtime
against the Isaac Lab ``TiledCamera`` output before training (see the migration
document section 8.3 / 8.4).
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .pick_multi_env import VisualWholeBodyPickMultiEnv
from .pick_multi_vision_env_cfg import VisualWholeBodyPickMultiVisionEnvCfg


class VisualWholeBodyPickMultiVisionEnv(VisualWholeBodyPickMultiEnv):
    """Vision student environment for B1+Z1 pick-multi."""

    cfg: VisualWholeBodyPickMultiVisionEnvCfg

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # table: STATIC collider (mirror ground-plane pattern; RigidObject table doesn't collide)
        self.cfg.table_cfg.spawn.func(
            self.cfg.table_cfg.prim_path,
            sim_utils.CuboidCfg(
                size=(0.6, 1.0, 0.25),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            translation=self.cfg.table_cfg.init_state.pos,
            orientation=self.cfg.table_cfg.init_state.rot,
        )
        self.object = RigidObject(self.cfg.object_cfg)
        self.front_camera = TiledCamera(self.cfg.front_camera)
        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.scene.sensors["front_camera"] = self.front_camera
        self.scene.sensors["wrist_camera"] = self.wrist_camera
        self.scene.rigid_objects["object"] = self.object
        light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # assign a semantic label to the graspable object so the semantic camera
        # can segment it into a binary target mask
        from isaaclab.sim.utils import find_matching_prim_paths

        from pxr import Semantics

        stage = self.scene.stage
        for prim_path in find_matching_prim_paths("/World/envs/env_.*/Object", stage=stage):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and not prim.HasAPI(Semantics.SemanticsAPI):
                Semantics.SemanticsAPI.Apply(prim)
            if prim:
                sem_api = Semantics.SemanticsAPI.Get(prim)
                sem_api.CreateSemanticTypeAttr().Set("class")
                sem_api.CreateSemanticDataAttr().Set("graspable")
        print("[VisionEnv] semantic label 'graspable' assigned to object prims")

    def _get_observations(self) -> dict:
        # derived state (base yaw, arm base, object distance …) needed by both channels
        self._update_derived_state()

        # --- teacher channel (1094) ---
        # Same as the state teacher: feature(1024) + robot(61) + action hist(9). Consumed
        # only by the DAgger labeler (skrl PPO), never by the student.
        robot_obs = self._compute_robot_observations()
        if self.cfg.last_commands:
            teacher_obs = torch.cat([robot_obs, self.command_history_buf[:, -1]], dim=-1)
        else:
            teacher_obs = torch.cat([robot_obs, self.action_history_buf[:, -1]], dim=-1)
        if not self.cfg.no_feature:
            teacher_obs = torch.cat([self.feature_obs, teacher_obs], dim=-1)

        # --- student channel (62269) ---
        # images (12x54x96) + a 61-dim proprioceptive tail. The tail must NOT leak the
        # object pose (robot_obs[0:6]) nor the base velocity (robot_obs[58:61]): the
        # original ``_compute_states_buf`` gives the student the robot obs minus those
        # privileged dims, plus the same action-history suffix the teacher sees.
        self.obtain_camera_obs()
        self.make_img_obs()
        images = self.camera_history_buf.view(self.num_envs, -1)
        if self.cfg.last_commands:
            history = self.command_history_buf[:, -1]
        else:
            history = self.action_history_buf[:, -1]
        student_proprio = torch.cat([robot_obs[:, 6:58], history], dim=-1)  # 52 + 9 = 61
        student_obs = torch.cat([images, student_proprio], dim=-1)
        self.obs_buf = torch.clamp(student_obs, -self.clip_obs, self.clip_obs)

        # update last-step buffers (inherited reward fns ``_reward_action_rate`` /
        # ``_reward_acc_penalty`` compare against these; without this they see zeros)
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.robot.data.joint_vel[:]

        # "policy" satisfies the DirectRLEnv observation-noise hook; the DAgger trainer
        # reads "states" (student) and "obs" (teacher) directly.
        return {"policy": self.obs_buf, "states": self.obs_buf, "obs": teacher_obs}

    def obtain_camera_obs(self):
        """Read the camera tensors and produce normalized depth / mask channels."""
        img_h, img_w = self.cfg.front_camera.height, self.cfg.front_camera.width

        # --- depth (world meters, positive forward)
        front_depth = self.front_camera.data.output["depth"].clone()
        wrist_depth = self.wrist_camera.data.output["depth"].clone()

        # --- semantic segmentation -> binary target mask
        front_seg = self.front_camera.data.output["semantic_segmentation"]
        wrist_seg = self.wrist_camera.data.output["semantic_segmentation"]
        front_mask = self._target_mask(front_seg)
        wrist_mask = self._target_mask(wrist_seg)

        # --- normalize depth into [0, 1] over [0, 2] m, invalid pixels -> 0
        front_depth = self._normalize_depth(front_depth)
        wrist_depth = self._normalize_depth(wrist_depth)

        # masked depth
        front_seg_depth = front_depth * front_mask
        wrist_seg_depth = wrist_depth * wrist_mask

        self.forward_mask = front_mask.view(self.num_envs, -1)
        self.wrist_mask = wrist_mask.view(self.num_envs, -1)
        self.forward_seg_depth = front_seg_depth.view(self.num_envs, -1)
        self.wrist_seg_depth = wrist_seg_depth.view(self.num_envs, -1)

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        # Isaac Lab TiledCamera depth is in meters (positive forward). Zero the
        # near-field (below the clip threshold) to match the original Isaac Gym
        # convention, then normalize the [0, 2] m range to [0, 1].
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = torch.clamp(depth, 0.0, 2.0)
        depth[depth < self.cfg.depth_clip_lower] = 0.0  # too close -> no signal
        return depth / 2.0

    def _target_mask(self, seg: torch.Tensor) -> torch.Tensor:
        """Convert the semantic segmentation tensor to a binary target mask."""
        # The semantic ids reported by the camera are environment-global; the
        # graspable object carries the 'graspable' class. We treat every non-zero
        # class id that is not the background (0) as the target for the first version.
        mask = (seg != 0).float()
        return mask

    def make_img_obs(self):
        tensor_obs = torch.cat(
            [self.forward_mask, self.wrist_mask, self.forward_seg_depth, self.wrist_seg_depth], dim=-1
        )
        self.camera_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([tensor_obs] * self.cfg.camera_history_len, dim=1),
            torch.cat([self.camera_history_buf[:, 1:], tensor_obs.unsqueeze(1)], dim=1),
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.camera_history_buf[env_ids] = 0.0

    def _init_buffers(self):
        super()._init_buffers()
        num_channels = 4
        self.camera_history_buf = torch.zeros(
            self.num_envs,
            self.cfg.camera_history_len,
            self.cfg.front_camera.width * self.cfg.front_camera.height * num_channels,
            device=self.device,
            dtype=torch.float,
        )
