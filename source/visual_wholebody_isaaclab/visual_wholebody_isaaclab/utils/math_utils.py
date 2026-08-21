# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Math helpers ported from the original ``visual_wholebody`` Isaac Gym project.

All quaternion functions here expect the **wxyz** convention used by Isaac Lab.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_conjugate, quat_mul


def torch_rand_float(lower, upper, shape, device) -> torch.Tensor:
    """Uniform random float tensor in ``[lower, upper)``."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def torch_rand_sign(shape, device) -> torch.Tensor:
    """Random +/-1 tensor."""
    return torch.randint(0, 2, size=shape, device=device) * 2 - 1


def torch_wrap_to_pi_minuspi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi)."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


@torch.jit.script
def sphere2cart(sphere_coords: torch.Tensor) -> torch.Tensor:
    """Convert spherical coordinates (l, pitch, yaw) to cartesian (x, y, z)."""
    l = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    cart_coords = torch.zeros_like(sphere_coords)
    cart_coords[:, 0] = l * torch.cos(pitch) * torch.cos(yaw)
    cart_coords[:, 1] = l * torch.cos(pitch) * torch.sin(yaw)
    cart_coords[:, 2] = l * torch.sin(pitch)
    return cart_coords


@torch.jit.script
def cart2sphere(cart_coords: torch.Tensor) -> torch.Tensor:
    """Convert cartesian coordinates (x, y, z) to spherical (l, pitch, yaw)."""
    sphere_coords = torch.zeros_like(cart_coords)
    xy_len = torch.norm(cart_coords[:, :2], dim=1)
    sphere_coords[:, 0] = torch.norm(cart_coords, dim=1)
    sphere_coords[:, 1] = torch.atan2(cart_coords[:, 2], xy_len)
    sphere_coords[:, 2] = torch.atan2(cart_coords[:, 1], cart_coords[:, 0])
    return sphere_coords


def orientation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Orientation error between two (wxyz) quaternions.

    This is a direct port of the original Isaac Gym ``orientation_error`` which
    returns the vector part of the error quaternion scaled by the sign of the
    scalar part (NOT the full axis-angle).
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 1:] * torch.sign(q_r[:, 0]).unsqueeze(-1)
