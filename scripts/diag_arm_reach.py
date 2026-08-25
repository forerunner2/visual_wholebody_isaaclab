"""实测：物体相对机械臂基座的距离 vs 手臂最大可达距离。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
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
    out = open("/tmp/diag_reach.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()

    arm_base = u.arm_base[0]
    obj = u.object_state[0]
    rel = obj[:3] - arm_base
    log(f"[初始] arm_base_world={arm_base.tolist()}")
    log(f"[初始] object_world={obj[:3].tolist()}")
    log(f"[初始] object相对arm_base(世界系)={rel.tolist()}  距离={rel.norm().item():.3f}m")

    # 用根姿态把相对位置转到局部系（z 方向就是"需要往下够多少"）
    root_quat = u.robot.data.root_quat_w[0]
    rel_local = quat_rotate_inverse(root_quat, rel)
    log(f"[初始] object相对arm_base(局部系)={rel_local.tolist()}")

    # 测手臂最大可达：把 EE 目标推到很远处，IK 驱动 100 步后测 EE 到 arm_base 的距离
    u.curr_ee_goal_cart[:] = torch.tensor([0.7, 0.0, 0.6], device=u.device)
    for _ in range(100):
        with torch.inference_mode():
            u._pre_physics_step(torch.zeros(u.num_envs, 9, device=u.device))
            for _ in range(u.cfg.decimation):
                u._apply_action()
                u.scene.write_data_to_sim()
                u.sim.step(render=False)
                u.scene.update(dt=u.physics_dt)
            u._update_derived_state()
    ee_to_base = torch.norm(u.ee_pos - u.arm_base, dim=-1)
    log(f"[全伸] EE到arm_base距离={ee_to_base[0].item():.3f}m  EE_world={u.ee_pos[0].tolist()}")
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
