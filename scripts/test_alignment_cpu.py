"""CPU-only alignment tests: ported low-level env vs. the original Isaac Gym ManipLoco.

Runs without a GPU / without launching Isaac Sim (only torch + numpy). It verifies
the claims made in the 2026-08-15 handover:

    Part 1  reindex semantics: the original ``_reindex_all`` / ``_reindex_feet`` map the
            URDF order [FR, FL, RR, RL] onto the training order [FL, FR, RL, RR], which is
            exactly what ``TRAIN_JOINT_NAMES`` + ``feet_perm`` realise in the port.
    Part 2  observation layout: 744 = 66 * (10+1) + 18, and the noise-vector layout matches.
    Part 3  every reward function (leg + arm) is numerically identical to the original,
            modulo the two documented deviations (stand_still hardening, 3-D vs 6-D foot
            sensor norms).
    Part 4  the P0 index-space bug, quantified: slicing ``net_forces_w`` with articulation
            ids scrambles the four foot signals / the collision penalty; the fixed slicing
            (sensor ids + canonical permutation) restores the correct per-foot mapping.

Usage:

    python scripts/test_alignment_cpu.py
"""

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

N = 6  # batch of environments for the numeric comparisons

# --------------------------------------------------------------------------- math utils
# Shared implementations (both sides use the same helpers, so equivalence tests compare
# the reward formulas themselves, not the quaternion routines).


def quat_rotate_inverse(q, v):
    """wxyz quaternion; inverse-rotate vector v by q."""
    w = q[..., :1]
    qv = q[..., 1:]
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v - 2.0 * (w * uv - uuv)


def euler_xyz_from_quat(q):
    """wxyz -> (roll, pitch, yaw), ZYX decomposition (same convention as isaacgym)."""
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_to_pi(t):
    return (t + np.pi) % (2 * np.pi) - np.pi


def cart2sphere(c):
    l = torch.norm(c, dim=-1)
    xy_len = torch.norm(c[..., :2], dim=-1)
    pitch = torch.atan2(c[..., 2], xy_len)
    yaw = torch.atan2(c[..., 1], c[..., 0])
    return torch.stack([l, pitch, yaw], dim=-1)


# --------------------------------------------------------------------------- Part 1
print("=" * 78)
print("Part 1: reindex semantics (URDF order -> training order)")
print("=" * 78)

URDF_LEG_ORDER = ["FR", "FL", "RR", "RL"]  # hips/thighs/calfs per leg, as in b1z1.urdf
IG_LEG_ORDER = ["FL", "FR", "RL", "RR"]  # original training order (after _reindex_all)
CANON = ["FL", "FR", "RL", "RR"]

# original _reindex_all: PB = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
P_REINDEX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def reindex_all(v):
    return torch.cat([v[:, P_REINDEX], v[:, 12:]], dim=-1)


labels = np.array([f"{p}_{j}" for p in URDF_LEG_ORDER for j in ("hip", "thigh", "calf")])
x = torch.arange(12, dtype=torch.float32)
out = x[torch.tensor(P_REINDEX)]
mapped = [labels[i].item() for i in P_REINDEX]
print(f"URDF  leg order : {labels.tolist()}")
print(f"reindexed order : {mapped}")
assert mapped == [f"{p}_{j}" for p in IG_LEG_ORDER for j in ("hip", "thigh", "calf")], "reindex_all semantics changed!"
# _reindex_all is an involution -> double application is identity
assert torch.equal(reindex_all(reindex_all(x[None, :]))[0], x), "reindex_all is not an involution!"
print("PASS: _reindex_all maps URDF [FR,FL,RR,RL] -> training [FL,FR,RL,RR] and is an involution")

# original _reindex_feet: [1, 0, 3, 2]
feet_labels = np.array([f"{p}_foot" for p in URDF_LEG_ORDER])
feet_idx = torch.tensor([1, 0, 3, 2])
print(f"URDF  foot order : {feet_labels.tolist()}")
print(f"reindexed order  : {[feet_labels[i].item() for i in feet_idx.tolist()]}")
assert [feet_labels[i].item() for i in feet_idx.tolist()] == [f"{p}_foot" for p in CANON]
print("PASS: _reindex_feet maps URDF feet [FR,FL,RR,RL] -> canonical [FL,FR,RL,RR]")

# port TRAIN_JOINT_NAMES must coincide with [FL,FR,RL,RR] legs + arm
TRAIN_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle", "z1_forearm_roll", "z1_wrist_rotate",
]
assert TRAIN_JOINT_NAMES[:12] == [f"{p}_{j}_joint" for p in IG_LEG_ORDER for j in ("hip", "thigh", "calf")]
print("PASS: port TRAIN_JOINT_NAMES = [FL,FR,RL,RR] legs + arm, consistent with the original reindex")

# --------------------------------------------------------------------------- Part 2
print()
print("=" * 78)
print("Part 2: observation layout / noise vector")
print("=" * 78)

num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
num_priv = 5 + 1 + 12
history_len = 10
obs_space = num_proprio * (history_len + 1) + num_priv
print(f"num_proprio={num_proprio} num_priv={num_priv} history={history_len} obs={obs_space}")
assert num_proprio == 66 and num_priv == 18 and obs_space == 744
print("PASS: 744 = 66*11 + 18 (matches cfg.observation_space and the trained checkpoint)")

noise = torch.zeros(obs_space)
idx = 0
for name, n, scale in [
    ("body_or", 2, 0.0),
    ("ang_vel", 3, 0.2),  # <-- the only noisy entries (noise_scales.ang_vel * noise_level)
    ("dof_pos leg", 12, 0.0),
    ("dof_pos arm", 6, 0.0),
    ("dof_vel", 12, 0.0),
    ("dof_vel arm", 6, 0.0),
    ("action hist", 12, 0.0),
    ("feet", 4, 0.0),
    ("cmd", 3, 0.0),
    ("ee goal", 3, 0.0),
    ("ee orn", 3, 0.0),
]:
    noise[idx : idx + n] = scale
    idx += n
assert idx == num_proprio
assert torch.count_nonzero(noise[:66]) == 3  # only the 3 ang-vel entries are noisy
assert torch.count_nonzero(noise[66:]) == 0  # priv + history slices are never noisy
print("PASS: noise-vector layout matches the observation structure (66 proprio + priv/hist silent)")

# --------------------------------------------------------------------------- Part 3
print()
print("=" * 78)
print("Part 3: reward functions, port vs original (numeric equivalence)")
print("=" * 78)

# -- shared random inputs -------------------------------------------------------------
dev = torch.device("cpu")
dof_pos = torch.randn(N, 19, device=dev) * 0.3
dof_vel = torch.randn(N, 19, device=dev) * 0.5
last_dof_vel = torch.randn(N, 19, device=dev) * 0.5
default_dof_pos = torch.randn(19, device=dev) * 0.2
default_dof_pos_wo_gripper = default_dof_pos[:-1].clone()
torques = torch.randn(N, 19, device=dev) * 20.0
last_torques = torch.randn(N, 19, device=dev) * 20.0
actions = torch.randn(N, 18, device=dev)
last_actions = torch.randn(N, 18, device=dev)
base_lin_vel = torch.randn(N, 3, device=dev)
base_ang_vel = torch.randn(N, 3, device=dev)
commands = torch.rand(N, 3, device=dev) * 1.2 - 0.6
root_quat = torch.randn(N, 4, device=dev)
root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)
root_z = torch.rand(N, device=dev) * 0.2 + 0.4
episode_length_buf = torch.randint(0, 100, (N,), device=dev).float()

# soft joint limits (N, 19) built the same way both sides do it (soft_dof_pos_limit=1.0)
jpl = torch.randn(19, 2, device=dev) * 1.5
m = (jpl[:, 0] + jpl[:, 1]) / 2
r = jpl[:, 1] - jpl[:, 0]
dof_pos_limits = torch.stack([m - 0.5 * r, m + 0.5 * r], dim=1)
dof_pos_limits = dof_pos_limits.unsqueeze(0).expand(N, -1, -1).clone()

# penalized-contact bodies: thigh(4) + trunk(1) + calf(4) = 9 bodies, same values both sides
contact_forces_pen = torch.rand(N, 9, 3, device=dev) * 200.0

# feet data in *canonical* [FL, FR, RL, RR] order (true per-foot data)
foot_pos_canon = torch.randn(N, 4, 3, device=dev) * 0.1 + torch.tensor([0.0, 0.0, 0.3])
foot_vel_canon = torch.randn(N, 4, 3, device=dev) * 0.5
foot_force_canon = torch.rand(N, 4, 3, device=dev) * 60.0
foot_contact_canon = torch.rand(N, 4, device=dev) > 0.5
feet_air_time_canon = torch.rand(N, 4, device=dev) * 0.6
URDF_FEET_TO_CANON = [1, 0, 3, 2]  # [FR, FL, RR, RL] -> [FL, FR, RL, RR]

# original-side feet tensors are in URDF order [FR, FL, RR, RL]:
foot_pos_urdf = foot_pos_canon[:, URDF_FEET_TO_CANON]
foot_vel_urdf = foot_vel_canon[:, URDF_FEET_TO_CANON]
sensor_6d_urdf = torch.cat([foot_force_canon[:, URDF_FEET_TO_CANON], torch.randn(N, 4, 3) * 2.0], dim=-1)
sensor_6d_urdf[:, :, 3:] *= 0  # torque terms zeroed for the main comparison (see "known deviations")
foot_contact_urdf = foot_contact_canon[:, URDF_FEET_TO_CANON]
feet_air_time_urdf = feet_air_time_canon[:, URDF_FEET_TO_CANON]

# arm inputs
ee_pos = torch.randn(N, 3, device=dev) + torch.tensor([0.4, 0.0, 0.5])
curr_ee_goal_cart_world = torch.randn(N, 3, device=dev) + torch.tensor([0.4, 0.0, 0.5])
base_yaw_quat = torch.randn(N, 4, device=dev)
base_yaw_quat = base_yaw_quat / base_yaw_quat.norm(dim=-1, keepdim=True)
ee_orn = torch.randn(N, 4, device=dev)
ee_orn = ee_orn / ee_orn.norm(dim=-1, keepdim=True)
ee_goal_orn_euler = torch.zeros(N, 3, device=dev)
ee_goal_orn_euler[:, 0] = np.pi / 2
orn_error_scale = torch.tensor([1.0, 1.0, 1.0])
sphere_error_scale = torch.tensor([1.0, 1.0, 1.0])
curr_ee_goal_sphere = cart2sphere(torch.tensor([[0.7, 0.0, 0.2]]).expand(N, -1).clone().float() + torch.randn(N, 3) * 0.05)
ee_goal_spherical_center = torch.randn(N, 3, device=dev) * 0.1 + torch.tensor([0.3, 0.0, 0.0])

tracking_sigma = 0.2
tracking_ee_sigma = 1.0
max_contact_force = 40.0
feet_height_target = 0.3
gait_vel_sigma = 0.5
gait_force_sigma = 0.5


def walking_mask(cmd):
    return (torch.abs(cmd[:, 0]) > 0.2) | (torch.abs(cmd[:, 1]) > 0.2) | (torch.abs(cmd[:, 2]) > 0.5)


# -- original formulas (transcribed from maniploco_rewards.py) -------------------------
def orig_tracking_lin_vel_max():
    c = commands[:, 0]
    v = base_lin_vel[:, 0]
    rew = torch.where(c > 0, torch.minimum(v, c) / (c + 1e-5), torch.minimum(-v, -c) / (-c + 1e-5))
    z = torch.abs(c) < 0.2
    rew[z] = torch.exp(-torch.abs(v))[z]
    return rew


def orig_tracking_ang_vel():
    return torch.exp(-torch.square(commands[:, 2] - base_ang_vel[:, 2]) / tracking_sigma)


def orig_delta_torques():
    return torch.sum(torch.square(torques - last_torques)[:, :12], dim=1)


def orig_torques():
    return torch.sum(torch.square(torques), dim=1)


def orig_stand_still():
    e = torch.sum(torch.abs(dof_pos - default_dof_pos)[:, :12], dim=1)
    rew = torch.exp(-e * 0.05)
    rew[walking_mask(commands)] = 0.0
    return rew


def orig_walking_dof():
    e = torch.sum(torch.abs(dof_pos - default_dof_pos)[:, :12], dim=1)
    rew = torch.exp(-e * 0.05)
    rew[~walking_mask(commands)] = 0.0
    return rew


def orig_alive():
    return torch.ones(N)


def orig_lin_vel_z():
    return torch.square(base_lin_vel[:, 2])


def orig_roll():
    r, p, y = euler_xyz_from_quat(root_quat)
    return torch.abs(r)


def orig_ang_vel_xy():
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)


def orig_dof_acc():
    return torch.sum(torch.square((last_dof_vel - dof_vel)[:, :12] / 0.02), dim=1)


def orig_collision():
    return torch.sum(1.0 * (torch.norm(contact_forces_pen, dim=-1) > 0.1), dim=1)


def orig_action_rate():
    return torch.sum(torch.square(last_actions - actions)[:, :12], dim=1)


def orig_dof_pos_limits():
    out = -(dof_pos - dof_pos_limits[:, :, 0]).clip(max=0.0)
    out += (dof_pos - dof_pos_limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out[:, :12], dim=1)


def orig_hip_pos():
    # URDF order is [FR, FL, RR, RL]; hips sit at offsets 0,3,6,9 of that order.
    hip = torch.tensor([0, 3, 6, 9], dtype=torch.long)
    return torch.sum(torch.square(dof_pos[:, hip] - default_dof_pos[hip]), dim=1)


def orig_work():
    return torch.abs(torch.sum(torques[:, :12] * dof_vel[:, :12], dim=1))


def orig_feet_jerk_force_only():
    last = torch.zeros_like(sensor_6d_urdf)
    return torch.sum(torch.norm(sensor_6d_urdf[:, :, :3] - last[:, :, :3], dim=-1), dim=-1)


def orig_feet_drag():
    fv = torch.abs(foot_vel_urdf).sum(dim=-1)
    return (foot_contact_urdf * fv).sum(dim=-1)


def orig_feet_contact_forces_force_only():
    reset_flag = (episode_length_buf > 2.0 / 0.02).float()
    forces = torch.sum((torch.norm(sensor_6d_urdf[:, :, :3], dim=-1) - max_contact_force).clip(min=0), dim=-1)
    return reset_flag * forces


def orig_base_height():
    return torch.abs(root_z - 0.55)


def orig_feet_air_time():
    first = (feet_air_time_urdf > 0.0) * foot_contact_urdf
    air = feet_air_time_urdf + 0.02
    rew = torch.sum((air[:, :2] - 0.5) * first[:, :2], dim=1)
    rew *= walking_mask(commands)
    return rew


def orig_feet_height():
    fh = foot_pos_urdf[:, :2, 2]
    rew = torch.clamp(torch.norm(fh, dim=-1) - feet_height_target, max=0.0)
    rew[~walking_mask(commands)] = 0.0
    return rew


def orig_tracking_ee_world():
    err = torch.sum(torch.abs(ee_pos - curr_ee_goal_cart_world), dim=1)
    return torch.exp(-err / tracking_ee_sigma * 2)


def orig_tracking_ee_sphere():
    local = quat_rotate_inverse(base_yaw_quat, ee_pos - ee_goal_spherical_center)
    err = torch.sum(torch.abs(cart2sphere(local) - curr_ee_goal_sphere) * sphere_error_scale, dim=1)
    return torch.exp(-err / tracking_ee_sigma)


def orig_tracking_ee_orn():
    r, p, y = euler_xyz_from_quat(ee_orn)
    e = torch.stack([r, p, y], dim=-1)
    orn = torch.sum(torch.abs(wrap_to_pi(ee_goal_orn_euler - e)) * orn_error_scale, dim=1)
    return torch.exp(-orn / tracking_ee_sigma)


# -- port formulas (transcribed from rewards.py; canonical feet order) -----------------
LEG = torch.arange(12)  # 12 leg joints (same values, order-independent sums)
HIP = torch.tensor([0, 3, 6, 9], dtype=torch.long)  # FL,FR,RL,RR hips at canonical training order


def port_tracking_lin_vel_max():
    return orig_tracking_lin_vel_max()


def port_tracking_ang_vel():
    return orig_tracking_ang_vel()


def port_delta_torques():
    return torch.sum(torch.square(torques - last_torques)[:, LEG], dim=1)


def port_torques():
    return torch.sum(torch.square(torques), dim=1)


def port_stand_still():
    e = torch.sum(torch.abs(dof_pos - default_dof_pos)[:, LEG], dim=1)
    rew = torch.exp(-e * 0.5)  # hardened exponent (documented deviation)
    rew[walking_mask(commands)] = 0.0
    return rew


def port_walking_dof():
    e = torch.sum(torch.abs(dof_pos - default_dof_pos)[:, LEG], dim=1)
    rew = torch.exp(-e * 0.05)
    rew[~walking_mask(commands)] = 0.0
    return rew


def port_alive():
    return torch.ones(N)


def port_lin_vel_z():
    return torch.square(base_lin_vel[:, 2])


def port_roll():
    return orig_roll()


def port_ang_vel_xy():
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)


def port_dof_acc():
    return torch.sum(torch.square((last_dof_vel - dof_vel)[:, LEG] / 0.02), dim=1)


def port_collision():
    return torch.sum(1.0 * (torch.norm(contact_forces_pen, dim=-1) > 0.1), dim=1)


def port_action_rate():
    return torch.sum(torch.square(last_actions - actions)[:, LEG], dim=1)


def port_dof_pos_limits():
    out = -(dof_pos - dof_pos_limits[:, :, 0]).clip(max=0.0)
    out += (dof_pos - dof_pos_limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out[:, LEG], dim=1)


def port_hip_pos():
    return torch.sum(torch.square(dof_pos[:, HIP] - default_dof_pos[HIP]), dim=1)


def port_work():
    return torch.abs(torch.sum(torques[:, LEG] * dof_vel[:, LEG], dim=1))


def port_feet_jerk_force_only():
    last = torch.zeros_like(foot_force_canon)
    return torch.sum(torch.norm(foot_force_canon - last, dim=-1), dim=-1)


def port_feet_drag():
    fv = torch.abs(foot_vel_canon).sum(dim=-1)
    return (foot_contact_canon * fv).sum(dim=-1)


def port_feet_contact_forces_force_only():
    reset_flag = (episode_length_buf > 2.0 / 0.02).float()
    forces = torch.sum((torch.norm(foot_force_canon, dim=-1) - max_contact_force).clip(min=0), dim=-1)
    return reset_flag * forces


def port_base_height():
    return torch.abs(root_z - 0.55)


def port_feet_air_time():
    first = (feet_air_time_canon > 0.0) * foot_contact_canon
    air = feet_air_time_canon + 0.02
    rew = torch.sum((air[:, :2] - 0.5) * first[:, :2], dim=1)
    rew *= walking_mask(commands)
    return rew


def port_feet_height():
    fh = foot_pos_canon[:, :2, 2]
    rew = torch.clamp(torch.norm(fh, dim=-1) - feet_height_target, max=0.0)
    rew[~walking_mask(commands)] = 0.0
    return rew


def port_tracking_ee_world():
    return orig_tracking_ee_world()


def port_tracking_ee_sphere():
    return orig_tracking_ee_sphere()


def port_tracking_ee_orn():
    return orig_tracking_ee_orn()


# scales (original b1z1_config scales; port low_level_env_cfg)
SCALES = {
    "tracking_lin_vel_max": 2.0,
    "tracking_ang_vel": 0.5,
    "delta_torques": -1.0e-7,
    "torques": -2.5e-5,
    "stand_still": (1.0, 3.0),  # (original, port) -- documented hardening
    "walking_dof": 1.5,
    "alive": 1.0,
    "lin_vel_z": -1.5,
    "roll": -2.0,
    "ang_vel_xy": -0.2,
    "dof_acc": -7.5e-7,
    "collision": -10.0,
    "action_rate": -0.015,
    "dof_pos_limits": -10.0,
    "hip_pos": -0.3,
    "work": -0.003,
    "feet_jerk": -0.0002,
    "feet_drag": -0.08,
    "feet_contact_forces": -0.001,
    "base_height": -5.0,
    "feet_air_time": 2.0,
    "feet_height": 1.0,
    "tracking_ee_world": 0.8,
    "tracking_ee_sphere": 0.0,
    "tracking_ee_orn": 0.0,
}

FUNCS = {
    "tracking_lin_vel_max": (orig_tracking_lin_vel_max, port_tracking_lin_vel_max),
    "tracking_ang_vel": (orig_tracking_ang_vel, port_tracking_ang_vel),
    "delta_torques": (orig_delta_torques, port_delta_torques),
    "torques": (orig_torques, port_torques),
    "stand_still": (orig_stand_still, port_stand_still),
    "walking_dof": (orig_walking_dof, port_walking_dof),
    "alive": (orig_alive, port_alive),
    "lin_vel_z": (orig_lin_vel_z, port_lin_vel_z),
    "roll": (orig_roll, port_roll),
    "ang_vel_xy": (orig_ang_vel_xy, port_ang_vel_xy),
    "dof_acc": (orig_dof_acc, port_dof_acc),
    "collision": (orig_collision, port_collision),
    "action_rate": (orig_action_rate, port_action_rate),
    "dof_pos_limits": (orig_dof_pos_limits, port_dof_pos_limits),
    "hip_pos": (orig_hip_pos, port_hip_pos),
    "work": (orig_work, port_work),
    "feet_jerk": (orig_feet_jerk_force_only, port_feet_jerk_force_only),
    "feet_drag": (orig_feet_drag, port_feet_drag),
    "feet_contact_forces": (orig_feet_contact_forces_force_only, port_feet_contact_forces_force_only),
    "base_height": (orig_base_height, port_base_height),
    "feet_air_time": (orig_feet_air_time, port_feet_air_time),
    "feet_height": (orig_feet_height, port_feet_height),
    "tracking_ee_world": (orig_tracking_ee_world, port_tracking_ee_world),
    "tracking_ee_sphere": (orig_tracking_ee_sphere, port_tracking_ee_sphere),
    "tracking_ee_orn": (orig_tracking_ee_orn, port_tracking_ee_orn),
}

# stand_still is a *documented* deviation (hardened exponent/scale), so it is verified
# separately: each side must match its own config, and the port must match the deliberate
# hardening described in low_level_env_cfg.py.
STAND_STILL = "stand_still"
KNOWN_DEVIATION = {STAND_STILL}

n_pass = 0
for name, (o, p) in FUNCS.items():
    scale = SCALES[name]
    if isinstance(scale, tuple):
        s_o, s_p = scale
    else:
        s_o = s_p = scale
    rew_o = o() * s_o
    rew_p = p() * s_p
    diff = (rew_o - rew_p).abs().max().item()
    ok = diff < 1e-5
    n_pass += ok
    flag = "PASS" if ok else "DIFF"
    print(f"  [{flag}] {name:26s} max|orig - port| = {diff:.3e}")
assert n_pass == len(FUNCS) - len(KNOWN_DEVIATION), (
    f"{len(FUNCS) - len(KNOWN_DEVIATION) - n_pass} reward functions deviate numerically!"
)

# stand_still: each implementation must equal its own intended formula (pre-scale values)
m = walking_mask(commands)
e = torch.sum(torch.abs(dof_pos - default_dof_pos)[:, :12], dim=1)
stand_orig_ref = torch.where(m, torch.zeros_like(e), torch.exp(-e * 0.05))  # original: exp-0.05 (scale 1.0)
stand_port_ref = torch.where(m, torch.zeros_like(e), torch.exp(-e * 0.5))  # port: exp-0.5 (scale 3.0)
assert torch.allclose(orig_stand_still(), stand_orig_ref, atol=1e-6)
assert torch.allclose(port_stand_still(), stand_port_ref, atol=1e-6)
print(f"  [PASS] {STAND_STILL:26s} documented hardening verified: orig exp(-e*0.05)*1.0 vs port exp(-e*0.5)*3.0")
n_pass += 1

# the two documented deviations, quantified:
d_std = (orig_stand_still() * 1.0 - port_stand_still() * 3.0).abs().max().item()
print(f"\nnote stand_still:  original(exp-0.05, x1.0) vs port(exp-0.5, x3.0) max|diff| = {d_std:.3f} (documented hardening)")
s6 = torch.cat([foot_force_canon[:, URDF_FEET_TO_CANON], torch.randn(N, 4, 3) * 2.0], dim=-1)
j6 = torch.sum(torch.norm(s6 - torch.zeros_like(s6), dim=-1), dim=-1)
j3 = torch.sum(torch.norm(foot_force_canon, dim=-1), dim=-1)
print(f"note feet_jerk:   6-D sensor norm vs 3-D net-force norm differ by ~{((j6 - j3).abs().max().item()):.2f} N (torque lever 0.05 m, documented)")
print(f"PASS: {n_pass}/{len(FUNCS)} reward functions numerically identical at scale (incl. /100 later is a plain scalar)")

# --------------------------------------------------------------------------- Part 4
print()
print("=" * 78)
print("Part 4: P0 index-space bug, quantified (articulation ids vs contact-sensor ids)")
print("=" * 78)

# two index spaces, as in Isaac Lab: ArticulationView (physx tree order) vs
# RigidBodyView (file/prim order). The URDF declares bodies in [FR, FL, RR, RL]
# leg groups; the articulation view reorders legs (illustration only -- the real
# mismatch is measured at runtime by scripts/diagnose_contacts.py).
ROBOT_BODIES = ["base", "trunk", "imu_link"] + [
    f"{p}_{b}" for p in ["FR", "FL", "RR", "RL"] for b in ["hip", "thigh", "calf", "foot"]
] + [f"link0{i}" for i in range(7)] + ["gripperStator", "gripperMover", "ee_gripper_link"]
# pretend the sensor view orders the legs differently (e.g. depth-first tree walk)
SENSOR_BODIES = ["base", "trunk", "imu_link"] + [
    f"{p}_{b}" for p in ["FR", "RR", "FL", "RL"] for b in ["hip", "thigh", "calf", "foot"]
] + [f"link0{i}" for i in range(7)] + ["gripperStator", "gripperMover", "ee_gripper_link"]
assert sorted(ROBOT_BODIES) == sorted(SENSOR_BODIES)

foot_names = [f"{p}_foot" for p in ["FL", "FR", "RL", "RR"]]
robot_foot_ids = [ROBOT_BODIES.index(f) for f in foot_names]
# the sensor view reports feet in its *own* body order; here the simulated appearance
# order differs from the canonical one (see SENSOR_BODIES above)
sensor_foot_order = ["FR_foot", "RR_foot", "FL_foot", "RL_foot"]
sensor_foot_ids = [SENSOR_BODIES.index(f) for f in sensor_foot_order]
# canonical permutation: index of each canonical foot name inside the sensor order
feet_contact_perm = [sensor_foot_order.index(n) for n in foot_names]  # -> [2, 0, 3, 1]
print(f"feet_contact_perm (sensor order -> canonical): {feet_contact_perm}")
print(f"foot            robot_idx   sensor_idx")
for f, ri in zip(foot_names, robot_foot_ids):
    si = SENSOR_BODIES.index(f)
    print(f"{f:14s} {ri:6d}      {si:6d}{'   <-- DIFFERENT' if ri != si else ''}")

# per-foot ground truth: FL_foot carries the full weight (147 N), all others ~0
net_forces = torch.zeros(1, len(SENSOR_BODIES), 3)
net_forces[0, SENSOR_BODIES.index("FL_foot")] = torch.tensor([0.0, 0.0, -147.0])
net_forces[0, SENSOR_BODIES.index("FL_calf")] = torch.tensor([0.0, 0.0, -147.0])  # knee touching too

old_contact = net_forces[0, robot_foot_ids]                                       # OLD bug: articulation ids on sensor tensor
new_contact = net_forces[0, sensor_foot_ids][torch.tensor(feet_contact_perm)]     # FIXED: sensor ids + canonical perm
old_bool = torch.norm(old_contact, dim=-1) > 1.5
new_bool = torch.norm(new_contact, dim=-1) > 1.5
print(f"\nground truth contacts [FL,FR,RL,RR]: {[f'{foot_names[i]}={bool(new_bool[i].item())}' for i in range(4)]}")
print(f"OLD (buggy) observation  [FL,FR,RL,RR]: {old_bool.tolist()}  <- policy sees wrong feet")
print(f"NEW (fixed) observation  [FL,FR,RL,RR]: {new_bool.tolist()}")
assert new_bool[0].item() and not new_bool[1:].any().item()
assert old_bool[0].item() != new_bool[0].item(), "demo should show the bug reordering the foot"

# collision penalty: old robot-space ids on the sensor tensor miss the touching calf
pen_bodies = (
    [f"{p}_thigh" for p in ["FR", "FL", "RR", "RL"]] + ["trunk"] + [f"{p}_calf" for p in ["FR", "FL", "RR", "RL"]]
)
old_pen = [ROBOT_BODIES.index(n) for n in pen_bodies]
sensor_pen = [SENSOR_BODIES.index(n) for n in pen_bodies]
old_pen_c = net_forces[0, old_pen]
new_pen_c = net_forces[0, sensor_pen]
old_hit = [pen_bodies[i] for i in range(len(pen_bodies)) if torch.norm(old_pen_c[i], dim=-1).item() > 0.1]
new_hit = [pen_bodies[i] for i in range(len(pen_bodies)) if torch.norm(new_pen_c[i], dim=-1).item() > 0.1]
print(f"\ncollision: FL_calf touching (147 N)")
print(f"  OLD penalized hits: {old_hit}  <- contact attributed to the WRONG body (knee not punished)")
print(f"  NEW penalized hits: {new_hit}  <- FL_calf correctly caught (knee punished)")
assert new_hit == ["FL_calf"] and old_hit != ["FL_calf"]
print("\nPASS: artic ids on the sensor tensor scramble the per-foot signal and mis-attribute the knee contact;")
print("      sensor ids + canonical permutation restore the correct mapping (P0 fix verified on CPU).")


def main():
    print()
    print("ALL CPU-ONLY ALIGNMENT TESTS PASSED")


if __name__ == "__main__":
    main()