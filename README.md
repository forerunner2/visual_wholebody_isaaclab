# Visual Whole-Body Control for Loco-Manipulation — Isaac Lab Port

[中文说明](README_CN.md)

This repository is an **unofficial Isaac Lab port** of the original
[Visual Whole-Body Control for Legged Loco-Manipulation](https://github.com/Ericonaldo/visual_whole_body)
project.

The original implementation was developed with NVIDIA Isaac Gym. This repository
migrates its environments, training pipeline, robot assets, and hierarchical
control architecture to **NVIDIA Isaac Lab 2.3.2**.

> This repository is an independent port and is not the official implementation
> maintained by the original authors. Please refer to the
> [original repository](https://github.com/Ericonaldo/visual_whole_body) and
> [project website](https://wholebody-b1.github.io/) for the original work.

## Overview

The project studies visual whole-body control for a Unitree B1 quadruped equipped
with a Z1 robotic arm. It uses a hierarchical control framework for
loco-manipulation tasks:

1. A low-level policy controls locomotion and tracks the target end-effector pose.
2. A state-based high-level teacher generates base and end-effector commands.
3. A vision-based student learns from the teacher through DAgger.

The implementation contains three main environments:

```text
Low-level environment
    Framework: Isaac Lab DirectRLEnv + custom RSL-RL
    Input:     base velocity command and end-effector target pose
    Output:    B1 leg actions and Z1 differential-IK targets

Pick-multi teacher environment
    Framework: Isaac Lab DirectRLEnv + SKRL PPO
    Input:     privileged robot, object, and table states
    Output:    end-effector delta, gripper command, and base command
    Note:      calls a frozen low-level policy

Pick-multi vision environment
    Framework: Isaac Lab DirectRLEnv + TiledCamera
    Input:     front/wrist depth images and semantic masks
    Output:    the same 9-dimensional high-level action
    Training:  DAgger visual student with a recurrent policy
```

## Main Changes from the Original Project

This port includes:

- Migration from NVIDIA Isaac Gym to NVIDIA Isaac Lab.
- Environments implemented with the Isaac Lab `DirectRLEnv` interface.
- B1 + Z1 robot configuration and assets adapted for Isaac Lab.
- Low-level whole-body controller migrated to a custom RSL-RL training stack.
- State-based high-level teacher training with SKRL PPO.
- Camera-based visual environment using Isaac Lab `TiledCamera`.
- DAgger training pipeline for the recurrent visual student.
- Preservation of the original hierarchical control frequency and policy structure.

Some behavior may differ from the original Isaac Gym implementation because of
differences in physics simulation, actuator models, contact handling, sensors,
and reinforcement-learning interfaces.

## Control Frequencies

The timing configuration follows the original project:

| Component | Configuration | Frequency |
|---|---:|---:|
| Physics simulation | `dt = 0.005 s` | 200 Hz |
| Low-level policy | Decimation 4 | 50 Hz |
| High-level policy | Decimation 32 | 6.25 Hz |
| Low-level episode | 10 seconds | 500 policy steps |
| High-level episode | 24 seconds | 150 policy steps |

Each high-level action is executed for eight low-level policy steps.

## Repository Structure

```text
visual_wholebody_isaaclab/
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/
│   │   ├── train_v1.py
│   │   ├── play_v1.py
│   │   ├── train.py
│   │   └── play.py
│   ├── skrl/
│   │   ├── train.py
│   │   └── play.py
│   └── dagger/
│       └── train_student.py
│
└── source/visual_wholebody_isaaclab/
    └── visual_wholebody_isaaclab/
        ├── assets/
        │   ├── b1_z1_cfg.py
        │   └── data/
        ├── learning/
        │   ├── low_level_policy.py
        │   ├── rsl_rl/
        │   └── dagger/
        └── tasks/direct/
            ├── low_level/
            └── pick_multi/
```

## Requirements

The current port was developed with:

- Ubuntu Linux
- NVIDIA GPU with a compatible driver
- NVIDIA Isaac Sim
- NVIDIA Isaac Lab 2.3.2
- Python environment configured by Isaac Lab
- PyTorch
- SKRL

Isaac Lab and Isaac Sim should be installed and verified before installing this
repository. Refer to the
[Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/) for the
official installation instructions.

## Installation

Clone this repository:

```bash
git clone https://github.com/forerunner2/visual_wholebody_isaaclab.git
cd visual_wholebody_isaaclab
```

Activate the Python environment used by Isaac Lab, then install the extension:

```bash
python -m pip install -e source/visual_wholebody_isaaclab
python -m pip install skrl
```

Verify that the environments are registered:

```bash
python scripts/list_envs.py
```

## Available Tasks

The repository registers the following main tasks:

```text
Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0
Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0
Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0
```

Use `scripts/list_envs.py` to check the exact task names available in the current
version.

## Training Workflow

### 1. Train the Low-Level Policy

The low-level controller uses the custom RSL-RL implementation ported from the
original project. It preserves the proprioceptive, privileged, and observation
history inputs, the dual leg/arm control heads, the privileged encoder, the state
history encoder, and the internal DAgger-style history update.

```bash
python scripts/rsl_rl/train_v1.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --num_envs 4096 --headless
```

Run a trained policy:

```bash
python scripts/rsl_rl/play_v1.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --checkpoint /path/to/model.pt
```

> The scripts in `scripts/rsl_rl/train.py` and the standard Isaac Lab RSL-RL
> configuration are retained as references. Checkpoints produced by the standard
> network are not compatible with the custom `train_v1.py` policy architecture.

### 2. Configure the Low-Level Checkpoint

Before training the high-level teacher, set `low_policy_path` in:

```text
source/visual_wholebody_isaaclab/visual_wholebody_isaaclab/
tasks/direct/pick_multi/pick_multi_env_cfg.py
```

Alternatively, use a command-line checkpoint argument when supported by the
corresponding script.

### 3. Train the State-Based High-Level Teacher

```bash
python scripts/skrl/train.py --task Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0 --num_envs 4096 --headless
```

The teacher receives privileged simulation states and produces a
nine-dimensional action containing:

- End-effector position delta
- End-effector orientation command
- Gripper command
- Mobile-base command

### 4. Train the Visual Student

After obtaining both the low-level policy and high-level teacher checkpoints,
train the visual student with DAgger:

```bash
python scripts/dagger/train_student.py --task Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0 --num_envs 64 --low_policy /path/to/low_level_model.pt --teacher_ckpt /path/to/teacher_checkpoint.pt
```

The student observes front-camera and wrist-camera depth/semantic information
and learns to imitate the state-based teacher.

## Notes

- A large number of parallel environments requires substantial GPU memory.
  Reduce `--num_envs` if an out-of-memory error occurs.
- Policies trained in the original Isaac Gym project may not be directly
  compatible with this Isaac Lab port.
- Checkpoint compatibility depends on the observation layout and network
  architecture used by the corresponding training script.
- This project is under active development. Training performance and exact
  reproduction of the original results are not guaranteed yet.
- Generated logs, checkpoints, caches, and experiment outputs should not be
  committed to Git.

## Original Project

This repository is derived from:

- Original repository:
  [Ericonaldo/visual_whole_body](https://github.com/Ericonaldo/visual_whole_body)
- Project website:
  [Visual Whole-Body Control](https://wholebody-b1.github.io/)
- Paper:
  [Visual Whole-Body Control for Legged Loco-Manipulation](https://arxiv.org/abs/2403.16967)

The original project also builds upon or refers to:

- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [SKRL](https://github.com/Toni-SM/skrl)
- [DeepWBC](https://github.com/MarkFzp/Deep-Whole-Body-Control)

## Acknowledgements

The primary research, original algorithm design, and original Isaac Gym
implementation were contributed by the authors and contributors of the
`visual_whole_body` project.

This repository focuses on adapting that work to the Isaac Lab framework. All
credit for the original method and research results belongs to the original
authors.

## Citation

If this repository is useful in your research, please cite the original paper:

```bibtex
@article{liu2024visual,
    title   = {Visual Whole-Body Control for Legged Loco-Manipulation},
    author  = {Liu, Minghuan and Chen, Zixuan and Cheng, Xuxin and
               Ji, Yandong and Yang, Ruihan and Wang, Xiaolong},
    journal = {arXiv preprint arXiv:2403.16967},
    year    = {2024}
}
```

When appropriate, please also link to this Isaac Lab port:

```text
https://github.com/forerunner2/visual_wholebody_isaaclab
```

## License

This repository contains code and assets adapted from upstream projects.
Please consult the licenses of the original project and all third-party
dependencies before using or redistributing the code, models, or assets.

