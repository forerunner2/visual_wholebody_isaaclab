"""诊断2：把机器人移到物体 0.4m 处（臂基座距物体0.4m），EE到位→合爪→抬升，看物体是否被夹起。"""
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
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg
import visual_wholebody_isaaclab.tasks  # noqa: F401

TASK = "Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0"

def main():
    out = open("/tmp/diag_g2.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    n = u.num_envs

    # 把机器人放到：身体在 env_origin + (-0.7, 0, 0.55)，面向 +x
    # → 臂基座在 env_origin + (-0.4, 0, 0.64)，物体在 env_origin + (0,0,0.273) → 臂基座距物体0.4m、低0.37m
    root = u.robot.data.default_root_state.clone()
    root[:, :3] = u.scene.env_origins + torch.tensor([-0.7, 0.0, 0.55], device=u.device)
    root[:, 3:7] = quat_from_euler_xyz(torch.zeros(n, device=u.device), torch.zeros(n, device=u.device), torch.zeros(n, device=u.device))
    u.robot.write_root_pose_to_sim(root[:, :7], torch.arange(n, device=u.device))
    u.scene.update(dt=u.physics_dt)
    u._update_derived_state()
    rel = u.object_state[0, :3] - u.arm_base[0]
    log(f"机器人已移动。物体相对臂基座={rel.tolist()} 距离={rel.norm().item():.3f}  (期望约[0.4,0,-0.37])")

    # 1) EE 目标 = 物体实际位置（相对臂基座，转局部系）
    from isaaclab.utils.math import quat_rotate_inverse
    goal = quat_rotate_inverse(u.robot.data.root_quat_w, u.object_state[:, :3] - u.arm_base)
    u.curr_ee_goal_cart[:] = goal
    log(f"EE目标(局部)={goal[0].tolist()}")
    for _ in range(60):
        u._pre_physics_step(torch.zeros(n, 9, device=u.device))
        for _ in range(u.cfg.decimation):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
        u._update_derived_state()
    d = torch.norm(u.ee_pos[0] - u.object_state[0, :3]).item()
    log(f"[到位] EE-物体距离={d:.3f}  物体z={u.object_state[0,2].item():.3f}")

    # 2) 合爪
    act = torch.zeros(n, 9, device=u.device); act[:, 6] = -1.0
    for _ in range(10):
        u._pre_physics_step(act)
        for _ in range(u.cfg.decimation):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
        u._update_derived_state()
    log(f"[合爪] 夹爪目标={u.gripper_dof_pos[0].item():.3f} 实际={u.robot.data.joint_pos[0, u.gripper_joint_ids[0]].item():.3f}  物体z={u.object_state[0,2].item():.3f}")

    # 3) 抬升：EE 目标 z 抬高 0.3
    u.curr_ee_goal_cart[:] = goal + torch.tensor([0.0, 0.0, 0.3], device=u.device)
    for _ in range(80):
        u._pre_physics_step(act)
        for _ in range(u.cfg.decimation):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
        u._update_derived_state()
    log(f"[抬升后] EEz={u.ee_pos[0,2].item():.3f}  物体z={u.object_state[0,2].item():.3f}  (物体z>0.3=被夹起)")
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
