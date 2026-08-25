"""可达性测试：命令 EE 目标到若干位置，看最终 EE 实际能到哪。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab_tasks.utils import parse_env_cfg
import visual_wholebody_isaaclab.tasks  # noqa: F401

TASK = "Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0"

def main():
    out = open("/tmp/diag_rm.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    n = u.num_envs
    obs, _ = env.reset()

    # 目标列表（臂基座局部系）：不同前伸 x、下探 z
    targets = [(0.30, 0.0, -0.37), (0.25, 0.0, -0.35), (0.35, 0.0, -0.30), (0.45, 0.0, -0.20)]
    for i, (tx, ty, tz) in enumerate(targets):
        # 每个 env 用不同目标（>= len 则循环）
        goal = torch.tensor([tx, ty, tz], device=u.device).repeat(n, 1)
        u.curr_ee_goal_cart[:] = goal
        for _ in range(40):
            u._pre_physics_step(torch.zeros(n, 9, device=u.device))
            for _ in range(u.cfg.decimation):
                u._apply_action()
                u.scene.write_data_to_sim()
                u.sim.step(render=False)
                u.scene.update(dt=u.physics_dt)
            u._update_derived_state()
        ee_local = quat_rotate_inverse(u.robot.data.root_quat_w, u.ee_pos - u.arm_base)
        err = torch.norm(ee_local - goal, dim=-1)
        log(f"目标({tx:.2f},{ty:.2f},{tz:.2f}): EE实际={ee_local[0].tolist()}  误差={err[0].item():.3f}")
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
