"""诊断：夹爪能否夹住并举起物体（把物体放到手臂可达处，合爪，抬升，看物体是否跟随）。"""
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
    out = open("/tmp/diag_grasp.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()

    # 把物体放到手臂可达处：arm_base + (0.35, 0, 0.25)，即末端当前高度附近
    n = u.num_envs
    obj_pos = u.arm_base.clone()
    obj_pos[:, 0] += 0.35
    obj_pos[:, 2] += 0.25
    obj_state = torch.zeros(n, 13, device=u.device)
    obj_state[:, :3] = obj_pos
    obj_state[:, 6] = 1.0
    u.object.write_root_pose_to_sim(obj_state[:, :7], torch.arange(n, device=u.device))
    u.scene.update(dt=u.physics_dt)
    u._update_derived_state()
    log(f"物体放到: {obj_pos[0].tolist()}  EE当前: {u.ee_pos[0].tolist()}  距离={torch.norm(u.ee_pos[0]-obj_pos[0]).item():.3f}")

    # 第一步：EE 目标指向物体（0.35 前、0.25 上，相对臂基座），让 IK 把末端送到物体
    u.curr_ee_goal_cart[:] = torch.tensor([0.35, 0.0, 0.25], device=u.device)
    def step_physics(steps):
        for _ in range(steps):
            u._pre_physics_step(torch.zeros(n, 9, device=u.device))
            for _ in range(u.cfg.decimation):
                u._apply_action()
                u.scene.write_data_to_sim()
                u.sim.step(render=False)
                u.scene.update(dt=u.physics_dt)
            u._update_derived_state()
    step_physics(40)
    log(f"[到位] EE={u.ee_pos[0].tolist()}  物体={u.object_state[0,:3].tolist()}  EE-物体距离={torch.norm(u.ee_pos[0]-u.object_state[0,:3]).item():.3f}")
    log(f"[到位] 夹爪目标位置={u.gripper_dof_pos[0].item():.3f} (下界-1.57闭合/上界0.0张开)")

    # 第二步：夹爪闭合（action[6]=-1），跑几步
    act = torch.zeros(n, 9, device=u.device); act[:, 6] = -1.0
    for _ in range(10):
        u._pre_physics_step(act)
        for _ in range(u.cfg.decimation):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
        u._update_derived_state()
    log(f"[合爪后] 夹爪实际位置={u.robot.data.joint_pos[0, u.gripper_joint_ids[0]].item():.3f}  物体z={u.object_state[0,2].item():.3f}")

    # 第三步：抬升（EE 目标 z 提到 0.5），看物体是否跟随
    u.curr_ee_goal_cart[:] = torch.tensor([0.35, 0.0, 0.5], device=u.device)
    for _ in range(60):
        u._pre_physics_step(act)
        for _ in range(u.cfg.decimation):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(dt=u.physics_dt)
        u._update_derived_state()
    log(f"[抬升后] EE={u.ee_pos[0].tolist()}  物体z={u.object_state[0,2].item():.3f}  物体速度z={u.object_state[0,9].item():.3f}")
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
