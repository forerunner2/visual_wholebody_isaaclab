# Visual Whole-Body(Isaac Lab)

Isaac Lab 2.3.2 port of the [visual_wholebody](https://github.com/Ericonaldo/visual_wholebody)
project (B1 quadruped + Z1 arm whole-body loco-manipulation).

## Architecture

```text
low-level env                       (DirectRLEnv, RSL-RL)
    input : base velocity cmd + EE target pose
    output: B1 leg RL actions + Z1 differential-IK targets

pick-multi teacher env              (DirectRLEnv, SKRL PPO, no cameras)
    input : object/table/robot ground truth
    output: EE delta + gripper + base command (9-dim), calls frozen low-level policy

pick-multi vision env               (DirectRLEnv, TiledCamera)
    input : front+wrist depth & semantic mask images
    output: same 9-dim action; DAgger/GRU student
```

## Time structure (preserved from the original)

```text
physics dt          0.005 s   (200 Hz)
low-level decimation  4       (50 Hz)
high-level decimation 32      (6.25 Hz, 8 low-level steps per high-level step)
episode length (low) 10 s     (500 low-level steps)
episode length (high) 24 s    (150 high-level steps)
```

## Layout

```text
scripts/
  list_envs.py
  rsl_rl/train.py, play.py           # low-level training
  skrl/train.py, play.py             # teacher training
  dagger/train_student.py            # DAgger visual student
source/visual_wholebody_isaaclab/visual_wholebody_isaaclab/
  assets/b1_z1_cfg.py, data/         # B1+Z1 URDF/meshes, obj_set
  tasks/direct/low_level/            # low-level env + rewards + rsl_rl cfg
  tasks/direct/pick_multi/           # teacher + vision env + skrl cfg
  learning/low_level_policy.py       # frozen low-level policy loader
  learning/dagger/                   # DAgger + GRU student (ported)
```

## Installation

```bash
conda activate env_isaaclab
cd /path/to/visual_wholebody_isaaclab
python -m pip install -e source/visual_wholebody_isaaclab
python -m pip install skrl
```

## Workflow

### 1. Verify the tasks are registered

```bash
python scripts/list_envs.py
```

### 2. Train the low-level policy (custom RSL-RL stack)

The low-level policy uses the **original custom RSL-RL stack** (ported from the
original project, NOT the stock Isaac Lab v3 workflow): custom `ActorCritic` with
prop/priv/hist observation split and dual control heads, `priv_encoder` +
`StateHistoryEncoder`, dual reward (leg/arm), `priv_reg` distillation and the
internal DAgger (`hist_encoding`) update loop.

```bash
python scripts/rsl_rl/train_v1.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --num_envs 4096 --headless
python scripts/rsl_rl/play.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --checkpoint <path>/model.pt
```

> The stock `rsl_rl/train.py` + `rsl_rl_ppo_cfg.py` is kept only as a reference for
> the standard Isaac Lab v3 workflow; it is NOT what the low-level policy is trained with.
> Old checkpoints trained with the stock network are incompatible — retrain with `train_v1.py`.

### 3. Train the high-level state teacher (SKRL)

Point `low_policy_path` in `pick_multi_env_cfg.py` at the low-level checkpoint, then:

```bash
python scripts/skrl/train.py --task Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0 --num_envs 4096 --headless
```

### 4. Train the visual student (DAgger)

```bash
python scripts/dagger/train_student.py --task Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0 --num_envs 64 --low_policy <low-level>/model.pt --teacher_ckpt <teacher>.pt
```
