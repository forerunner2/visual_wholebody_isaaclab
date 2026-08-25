"""临时诊断：教师环境每步的奖励 / 终止原因 / 物体&机器人状态。

用法:
  python scripts/diagnose_teacher_term.py --num_envs 8 --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse
from isaaclab_tasks.utils import parse_env_cfg

import visual_wholebody_isaaclab.tasks  # noqa: F401

TASK = "Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0"


def main():
    out = open("/tmp/diag_result.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()

    log(f"num_envs={u.num_envs}  dt={u.dt}  max_episode_length={u.max_episode_length}")
    log(f"init_ee_goal_cart[0]={u.curr_ee_goal_cart[0].tolist()}")
    log(f"table_height[0]={u.table_height[0].item():.3f}  obj_height[0]={u.obj_height[0].item():.3f}")

    # 首帧打印关键帧量，定位 ik_fail 的 Z 差
    ee_local0 = quat_rotate_inverse(u.robot.data.root_quat_w, u.ee_pos - u.arm_base)
    log(f"[首帧] goal_z={u.curr_ee_goal_cart[0, -1].item():.3f}  ee_local_z={ee_local0[0, -1].item():.3f}  "
        f"|diff|={(u.curr_ee_goal_cart[0, -1] - ee_local0[0, -1]).abs().item():.3f}")
    log(f"[首帧] ee_pos_world[0]={u.ee_pos[0].tolist()}  arm_base_world[0]={u.arm_base[0].tolist()}  "
        f"root_pos[0]={u.robot.data.root_pos_w[0].tolist()}")

    for step in range(15):
        actions = torch.zeros(u.num_envs, 9, device=u.device)
        obs, rew, term, trunc, info = env.step(actions)

        cube_z = u.object_state[:, 2]
        robot_z = u.robot.data.root_pos_w[:, 2]
        roll, pitch, _ = euler_xyz_from_quat(u.robot.data.root_quat_w)
        ee_local = quat_rotate_inverse(u.robot.data.root_quat_w, u.ee_pos - u.arm_base)
        ik_fail = (u.curr_ee_goal_cart[:, -1:] - ee_local[:, -1:]).norm(dim=-1) > 0.2

        log(
            f"step {step:2d}: rew_mean={rew.mean().item():.4f}  "
            f"term={term.sum().item()} trunc={trunc.sum().item()}  "
            f"cube_z[min={cube_z.min().item():.3f},mean={cube_z.mean().item():.3f}]  "
            f"robot_z[min={robot_z.min().item():.3f}]  "
            f"ikfail={ik_fail.sum().item()} cubefalls={(cube_z < 0.25).sum().item()} "
            f"zterm={(robot_z < 0.1).sum().item()} rterm={(roll.abs() > 0.8).sum().item()} "
            f"pterm={(pitch.abs() > 0.8).sum().item()} pitch_mean={pitch.mean().item():.3f} roll_mean={roll.mean().item():.3f}"
        )

    env.close()
    out.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
