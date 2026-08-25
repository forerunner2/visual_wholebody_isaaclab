"""检查 table prim 是否存在、有无碰撞。"""
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
    out = open("/tmp/diag_table.txt", "w")
    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        out.write(line + "\n"); out.flush()

    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    stage = u.scene.stage
    from pxr import UsdPhysics
    found = 0
    for i in range(args_cli.num_envs):
        p = f"/World/envs/env_{i}/Table"
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            found += 1
            apis = []
            for c in prim.GetAllChildren():
                for gc in c.GetAllChildren():
                    for api in gc.GetAppliedSchemas():
                        apis.append(str(gc.GetPath()) + ":" + api)
            log(f"{p} 有效, 子prim schemas: {apis[:6]}")
        else:
            log(f"{p} 不存在")
    log(f"Table prims found: {found}/{args_cli.num_envs}")
    # object
    op = stage.GetPrimAtPath("/World/envs/env_0/Object")
    log(f"Object prim children: {[str(c.GetPath()) for c in op.GetAllChildren()]}")
    bl = stage.GetPrimAtPath("/World/envs/env_0/Object/baseLink")
    log(f"baseLink schemas: {[a.GetTypeName() for a in bl.GetAppliedSchemas()]}")
    for c in bl.GetAllChildren():
        log(f"  baseLink child {str(c.GetPath())}: {[a.GetTypeName() for a in c.GetAppliedSchemas()]}")
    # table mesh 位置
    tm = stage.GetPrimAtPath("/World/envs/env_0/Table")
    from pxr import UsdGeom
    xf = UsdGeom.Xformable(tm)
    print("table xform ops:", xf.GetOrderedXformOps())
    env.close()
    out.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
