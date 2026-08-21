"""Headless smoke test for the low-level B1+Z1 environment (milestone A/B)."""

import faulthandler
import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# dump traceback if we hang (do not exit; first URDF conversion can be slow)
faulthandler.dump_traceback_later(300)

import gymnasium as gym
import visual_wholebody_isaaclab.tasks  # noqa: F401
from visual_wholebody_isaaclab.tasks.direct.low_level.low_level_env_cfg import VisualWholeBodyLowLevelEnvCfg


def main():
    print("starting env creation", flush=True)
    env_cfg = VisualWholeBodyLowLevelEnvCfg()
    env_cfg.scene.num_envs = 8
    # keep observe_priv=True for the 744-dim obs, disable physx-touching randomization
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False
    env = gym.make("Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0", cfg=env_cfg)
    env = env.unwrapped
    print("env created", flush=True)

    print("=" * 80)
    print("num_envs:", env.num_envs)
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("num_actions:", env.num_actions)
    print("obs dim:", env.cfg.observation_space)

    robot = env.robot
    print("\nrobot.num_joints:", robot.num_joints)
    print("robot.joint_names:", robot.joint_names)
    print("\nrobot.num_bodies:", robot.num_bodies)
    print("robot.body_names:", robot.body_names)

    print("\nleg_joint_ids:", env.leg_joint_ids)
    print("arm_joint_ids:", env.arm_joint_ids)
    print("gripper_joint_ids:", env.gripper_joint_ids)
    print("ee_body_id:", env.ee_body_id)
    print("wrist_body_id:", env.wrist_body_id)
    print("foot_body_ids:", env.foot_body_ids)
    print("penalized_contact_body_ids:", env.penalized_contact_body_ids)
    print("hip_indices:", env.hip_indices.tolist())

    print("\ndefault joint pos:", env.robot.data.default_joint_pos)

    # jacobian shape check
    jac = robot.root_physx_view.get_jacobians()
    print("\njacobian shape:", tuple(jac.shape))

    obs, _ = env.reset()
    print("\nobs keys:", list(obs.keys()))
    print("policy obs shape:", obs["policy"].shape)
    assert obs["policy"].shape == (8, 744), f"expected (8, 744), got {obs['policy'].shape}"

    # step with zero actions
    for i in range(10):
        obs, rew, terminated, truncated, info = env.step(torch.zeros(8, env.num_actions, device=env.device))
        assert not torch.isnan(rew).any(), f"NaN reward at step {i}"
        assert not torch.isnan(obs["policy"]).any(), f"NaN obs at step {i}"
        if i % 5 == 0:
            print(f"step {i}: reward mean={rew.mean().item():.4f} term={terminated.sum().item()} trunc={truncated.sum().item()}")
            print("  root z:", env.robot.data.root_pos_w[:, 2].mean().item())
            print("  ee pos:", env.ee_pos[0].tolist())
            print("  ee goal world:", env.curr_ee_goal_cart_world[0].tolist())

    print("\nSMOKE TEST PASSED")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
