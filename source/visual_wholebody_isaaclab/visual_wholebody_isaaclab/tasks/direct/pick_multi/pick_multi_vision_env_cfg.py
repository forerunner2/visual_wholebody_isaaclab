# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the visual (camera) student environment.

This is the TiledCamera version of the pick-multi task. The visual student
observation is composed of (see the original ``B1Z1Base._get_camera_obs``):

    forward depth, wrist depth, forward target mask, wrist target mask,
    forward masked depth, wrist masked depth

assembled into 4 channels (mask, mask, masked depth, masked depth) with a
history length of 3 -> 12 channels per image.
"""

from __future__ import annotations

import math

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

from .pick_multi_env_cfg import VisualWholeBodyPickMultiEnvCfg

# camera offset quaternions (wxyz), recomputed from the original Euler angles.
# front: rotation [0,0,0] -> identity
# wrist: gymapi.from_euler_zyx(z=-1.57, y=0, x=-0.87) = Rz(-1.57)Ry(0)Rx(-0.87)
FRONT_CAM_QUAT = (1.0, 0.0, 0.0, 0.0)
WRIST_CAM_QUAT = (0.641509, -0.298101, 0.297863, -0.640999)

# approximate pinhole params for 69 deg horizontal FOV at 96 px wide
_FOCAL_LENGTH = 11.0
_HORIZONTAL_APERTURE = 2.0 * _FOCAL_LENGTH * math.tan(math.radians(69.0) / 2.0)


def _pinhole_cam() -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(
        focal_length=_FOCAL_LENGTH,
        horizontal_aperture=_HORIZONTAL_APERTURE,
        clipping_range=(0.05, 5.0),
    )


def _front_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/front_camera",
        data_types=["depth", "semantic_segmentation"],
        width=96,
        height=54,
        spawn=_pinhole_cam(),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.425, 0.04, 0.12), rot=FRONT_CAM_QUAT),
    )


def _wrist_camera_cfg() -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/link06/wrist_camera",
        data_types=["depth", "semantic_segmentation"],
        width=96,
        height=54,
        spawn=_pinhole_cam(),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0955, 0.22, -0.03175), rot=WRIST_CAM_QUAT),
    )


class VisualWholeBodyPickMultiVisionEnvCfg(VisualWholeBodyPickMultiEnvCfg):
    """Vision student config for the B1+Z1 pick-multi task."""

    # DAgger needs both channels out of one env:
    #   - ``observation_space`` = teacher state obs (feature 1024 + robot 61 + action hist 9 = 1094)
    #   - ``state_space``      = student vision obs (12x54x96 images + 61-dim proprio tail = 62269)
    # So the object feature must be loaded; it is only ever fed to the teacher (see
    # ``VisualWholeBodyPickMultiVisionEnv._get_observations``). ``observation_space`` is
    # inherited from the parent (1094) via ``__post_init__``.
    no_feature: bool = False

    front_camera: TiledCameraCfg = MISSING  # type: ignore
    wrist_camera: TiledCameraCfg = MISSING  # type: ignore
    camera_history_len: int = 3
    depth_clip_lower: float = 0.15
    img_delay_frame: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.front_camera = _front_camera_cfg()
        self.wrist_camera = _wrist_camera_cfg()
        # image obs = 96*54*12, plus the small proprioception tail for the teacher states
        self.state_space = 96 * 54 * 12 + 61
