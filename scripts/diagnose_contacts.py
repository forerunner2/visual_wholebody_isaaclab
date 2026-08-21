"""Diagnose foot/penalized contact indexing between the Articulation and ContactSensor.

Prints the two body-name lists (articulation vs contact sensor) and the index
each of the four feet resolves to in each space. If the two indices differ for
the same foot name, the contact tensor was being mis-indexed before the fix.

Run headless:

    python scripts/diagnose_contacts.py [--steps 20]

No checkpoint required (it steps with zero actions, which is enough to confirm
ordering and see the default crouch contact pattern).
"""

import argparse
import faulthandler

import torch

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

faulthandler.dump_traceback_later(300)

import gymnasium as gym
import visual_wholebody_isaaclab.tasks  # noqa: F401
from visual_wholebody_isaaclab.tasks.direct.low_level.low_level_env_cfg import VisualWholeBodyLowLevelEnvCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run with the CPU physics backend (sim.device='cpu'); useful on machines without a usable NVIDIA driver.",
    )
    args = parser.parse_args()

    env_cfg = VisualWholeBodyLowLevelEnvCfg()
    env_cfg.scene.num_envs = 4
    if args.cpu:
        env_cfg.sim.device = "cpu"
        env_cfg.sim.use_fabric = True  # fabric is a data channel, works with the CPU backend
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False
    env = gym.make("Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0", cfg=env_cfg).unwrapped

    print("=" * 80)
    print("sim device              :", env_cfg.sim.device)
    print("robot.num_bodies        :", env.robot.num_bodies)
    print("contact_sensor.num_bodies:", env.contact_sensor.num_bodies)
    print()
    print("robot.body_names        :", env.robot.body_names)
    print("contact_sensor.body_names:", env.contact_sensor.body_names)
    print()
    print("foot_body_names (robot)  :", env.foot_body_names)
    print("foot_contact_names(sensor):", env.foot_contact_names)
    print("foot_body_ids (robot)    :", env.foot_body_ids)
    print("foot_contact_ids (sensor):", env.foot_contact_ids)
    print("feet_perm          :", env.feet_perm.tolist())
    print("feet_contact_perm  :", env.feet_contact_perm.tolist())
    print("penalized_contact_body_ids (sensor):", env.penalized_contact_body_ids)

    # Show, foot by foot, the ABSOLUTE body index each name resolves to in both index
    # spaces. (The relative order inside the two 4-element foot lists happens to be the
    # same, but the absolute body indices -- the values actually used to slice
    # ``body_pos_w`` and ``net_forces_w`` -- must come from the matching space.)
    foot_order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    print()
    print("foot   robot_idx   sensor_idx   (absolute body indices into body_pos_w / net_forces_w)")
    mismatch = False
    for name in foot_order:
        r = env.foot_body_ids[env.foot_body_names.index(name)]
        s = env.foot_contact_ids[env.foot_contact_names.index(name)]
        flag = "" if r == s else "   <-- DIFFERENT"
        if r != s:
            mismatch = True
        print(f"{name:8s} {r:8d}  {s:10d}{flag}")
    print()
    print("MISMATCH (absolute indices differ between spaces):", mismatch)
    if mismatch:
        print(
            "=> slicing net_forces_w with robot indices was wrong; the fixed code uses "
            "foot_contact_ids (sensor space) + feet_contact_perm."
        )

    # Run a few steps with zero actions and inspect contact norms / foot heights.
    obs, _ = env.reset()
    for i in range(args.steps):
        obs, rew, term, trunc, info = env.step(torch.zeros(4, env.num_actions, device=env.device))
        if i == args.steps - 1:
            foot_forces = torch.norm(env.foot_contact_forces, dim=-1)[0]  # canonical [FL,FR,RL,RR]
            foot_z = env.foot_positions[0, :, 2]
            contacts = env.foot_contacts_from_sensor[0]
            pen = torch.norm(env.contact_forces[:, env.penalized_contact_body_ids], dim=-1).max().item()
            print("\n=== final step (env 0), canonical [FL, FR, RL, RR] ===")
            print("foot contact norm :", [f"{v:.3f}" for v in foot_forces.tolist()])
            print("foot contact bool :", contacts.tolist())
            print("foot z (height)   :", [f"{v:.3f}" for v in foot_z.tolist()])
            print("root z            :", f"{env.robot.data.root_pos_w[0, 2].item():.3f}")
            print("max penalized-body contact norm (thigh/trunk/calf):", f"{pen:.3f}")

    env.close()
    print("\nDONE")


if __name__ == "__main__":
    main()
    simulation_app.close()