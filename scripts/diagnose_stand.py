"""Diagnose why the dog leans forward / fails to stand.

Loads a trained low-level checkpoint and runs a few hundred steps, reporting
body height, pitch, contact pattern and episode survival. Run with the trained
checkpoint path as argv[1].
"""

import faulthandler
import sys

import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

faulthandler.dump_traceback_later(400)

import gymnasium as gym
import visual_wholebody_isaaclab.tasks  # noqa: F401
from visual_wholebody_isaaclab.tasks.direct.low_level.low_level_env_cfg import VisualWholeBodyLowLevelEnvCfg
from visual_wholebody_isaaclab.learning.low_level_policy import LowLevelPolicy


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    env_cfg = VisualWholeBodyLowLevelEnvCfg()
    env_cfg.scene.num_envs = 8
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False
    env = gym.make("Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0", cfg=env_cfg).unwrapped
    print("env created", flush=True)

    policy = None
    if ckpt:
        policy = LowLevelPolicy(ckpt, obs_space=744, action_space=18, device=env.device)

    obs, _ = env.reset()
    print(f"initial root pos z: {env.robot.data.root_pos_w[:,2].mean().item():.3f}", flush=True)
    print(f"default joint pos (first 12): {env.default_dof_pos[:12].tolist()}", flush=True)

    heights, pitches, ep_lens = [], [], 0
    feet_contact_counts = torch.zeros(4)
    n_steps = 300
    for i in range(n_steps):
        if policy is not None:
            low_obs = env._get_observations()["policy"].detach()
            acts = policy.act(low_obs, hist_encoding=True)
            acts[:, 12:] = 0.0
        else:
            acts = torch.zeros(8, env.num_actions, device=env.device)
        obs, rew, terminated, truncated, info = env.step(acts)
        z = env.robot.data.root_pos_w[:, 2].mean().item()
        roll, pitch, yaw = env.robot.data.root_quat_w[:, 0], env.robot.data.root_quat_w[:, 1], env.robot.data.root_quat_w[:, 2]
        # wxyz -> euler approx for pitch via tensor
        import math
        pitch_deg = torch.rad2deg(2 * torch.atan2(pitch, env.robot.data.root_quat_w[:, 3])).mean().item()
        heights.append(z)
        pitches.append(pitch_deg)
        ep_lens += 1
        if i < 50:
            feet_contact_counts += env.foot_contacts_from_sensor.float().mean(0).cpu()
    print(f"\n=== after {n_steps} steps ===", flush=True)
    print(f"body height: last={heights[-1]:.3f} mean={sum(heights)/len(heights):.3f}", flush=True)
    print(f"body pitch (deg, + = forward tilt): last={pitches[-1]:.2f} mean={sum(pitches)/len(pitches):.2f}", flush=True)
    print(f"survived steps: {ep_lens}", flush=True)
    print(f"feet contact (first 50 steps, per foot [FL,FR,RL,RR]): {feet_contact_counts.tolist()}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
