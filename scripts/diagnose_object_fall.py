"""临时诊断：物体在单步物理后（未 reset）落在哪里，判断是否漏穿桌面。"""
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
from isaaclab_tasks.utils import parse_env_cfg
import visual_wholebody_isaaclab.tasks  # noqa: F401

TASK = "Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0"


def main():
    out = open("/tmp/diag_object.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    log(f"初始 object_z={u.object.data.root_state_w[:, 2].tolist()}  table_z_center={u.table_z_center}  table_half={u.table_half_height}")

    actions = torch.zeros(u.num_envs, 9, device=u.device)
    # 手动跑一个完整高层步（32 子步）但不走 _get_dones/_reset_idx
    u._pre_physics_step(actions)
    for _ in range(u.cfg.decimation * 6):
        u._apply_action()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
    log(f"物理后 object_z={u.object.data.root_state_w[:, 2].tolist()}")
    log(f"物理后 object_vel_z={u.object.data.root_state_w[:, 9].tolist()}")
    env.close()
    out.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
