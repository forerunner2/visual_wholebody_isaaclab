"""聚焦测试：合爪指令是否真正写入 joint_pos_target 并驱动夹爪。"""
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
from isaaclab_tasks.utils import parse_env_cfg
import visual_wholebody_isaaclab.tasks  # noqa: F401

TASK = "Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0"

def main():
    out = open("/tmp/diag_gcmd.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    gid = u.gripper_joint_ids[0]
    log(f"gripper index={gid}  初始 joint_pos_target={u.robot.data.joint_pos_target[0, gid].item():.3f}  joint_pos={u.robot.data.joint_pos[0, gid].item():.3f}")

    act = torch.zeros(u.num_envs, 9, device=u.device); act[:, 6] = -1.0
    u._pre_physics_step(act)
    log(f"pre_physics后: actions[6]={u.actions[0,6].item():.3f}  gripper_dof_pos={u.gripper_dof_pos[0].item():.3f}")
    # 隔离测试 _set_gripper
    u.actions[:] = 0.0; u.actions[:, 6] = -1.0
    u._set_gripper()
    log(f"隔离_set_gripper(close): gripper_dof_pos={u.gripper_dof_pos[0].item():.3f}  (期望-1.571)")
    u.actions[:, 6] = 1.0
    u._set_gripper()
    log(f"隔离_set_gripper(open): gripper_dof_pos={u.gripper_dof_pos[0].item():.3f}  (期望0.0)")
    u._apply_action()
    log(f"apply_action后: joint_pos_target={u.robot.data.joint_pos_target[0, gid].item():.3f}")
    for k in range(8):
        u._pre_physics_step(act)
        for _ in range(u.cfg.decimation):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
        u._update_derived_state()
        log(f"step{k}: pos_target={u.robot.data.joint_pos_target[0, gid].item():.3f}  joint_pos={u.robot.data.joint_pos[0, gid].item():.3f}")
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
