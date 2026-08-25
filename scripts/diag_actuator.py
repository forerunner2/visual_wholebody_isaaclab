"""检查各 actuator 覆盖的关节，确认夹爪是否被接管。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import visual_wholebody_isaaclab.tasks  # noqa: F401

TASK = "Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0"

def main():
    out = open("/tmp/diag_act.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    log(f"关节总数: {u.robot.num_joints}")
    log(f"关节名: {u.robot.joint_names}")
    for name, act in u.robot.actuators.items():
        idx = act.joint_indices.tolist()
        log(f"actuator[{name}]: indices={idx} names={[u.robot.joint_names[i] for i in idx]}")
    # 夹爪目标流向检查
    gid = u.gripper_joint_ids
    log(f"gripper_joint_ids={gid}")
    log(f"data.joint_pos_target[:, gid]={u.robot.data.joint_pos_target[0, gid].item():.3f}")
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
