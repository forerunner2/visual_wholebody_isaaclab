# 面向移动操作的视觉全身控制——Isaac Lab 移植版

[English](README.md)

本仓库是
[Visual Whole-Body Control for Legged Loco-Manipulation](https://github.com/Ericonaldo/visual_whole_body)
项目的一个**非官方 Isaac Lab 移植版本**。

原项目基于 NVIDIA Isaac Gym 开发。本仓库将其强化学习环境、训练流程、
机器人资源和分层控制架构迁移到了 **NVIDIA Isaac Lab 2.3.2**。

> 本仓库由社区独立移植，并非原作者维护的官方实现。有关原始研究与实现，
> 请访问[原始代码仓库](https://github.com/Ericonaldo/visual_whole_body)
> 和[项目主页](https://wholebody-b1.github.io/)。

## 项目简介

本项目研究搭载 Z1 机械臂的 Unitree B1 四足机器人视觉全身控制，
目标是让机器人完成移动操作任务。系统采用分层控制架构：

1. 低层策略负责机器人移动，并跟踪末端执行器目标位姿。
2. 基于状态的高层教师策略生成机器人底盘和末端执行器指令。
3. 基于视觉的学生策略通过 DAgger 模仿高层教师策略。

当前实现主要包含以下三个环境：

```text
低层控制环境
    框架：Isaac Lab DirectRLEnv + 自定义 RSL-RL
    输入：底盘速度指令、末端执行器目标位姿
    输出：B1 腿部动作、Z1 差分逆运动学目标

Pick-Multi 高层教师环境
    框架：Isaac Lab DirectRLEnv + SKRL PPO
    输入：机器人、物体和桌子的特权状态信息
    输出：末端位姿增量、夹爪指令、底盘指令
    说明：调用冻结的低层策略执行动作

Pick-Multi 视觉学生环境
    框架：Isaac Lab DirectRLEnv + TiledCamera
    输入：前置相机和腕部相机的深度图与语义分割图
    输出：与教师策略相同的 9 维高层动作
    训练：使用 DAgger 训练带循环网络的视觉策略
```

## 相比原项目的主要修改

本移植版本主要完成了以下工作：

- 将仿真平台从 NVIDIA Isaac Gym 迁移到 NVIDIA Isaac Lab。
- 使用 Isaac Lab 的 `DirectRLEnv` 接口重新实现环境。
- 将 B1 + Z1 机器人配置和资源适配到 Isaac Lab。
- 将原始低层全身控制策略迁移到自定义 RSL-RL 训练框架。
- 使用 SKRL PPO 训练基于状态的高层教师策略。
- 使用 Isaac Lab `TiledCamera` 构建视觉环境。
- 移植基于 DAgger 的循环视觉学生策略训练流程。
- 尽量保留原项目的分层控制频率和策略网络结构。

由于 Isaac Gym 与 Isaac Lab 在物理仿真、执行器模型、接触处理、传感器和
强化学习接口等方面存在差异，本项目的运行结果可能与原项目不完全一致。

## 控制频率

当前配置保留了原项目的时间结构：

| 模块 | 配置 | 频率 |
|---|---:|---:|
| 物理仿真 | `dt = 0.005 s` | 200 Hz |
| 低层策略 | Decimation 4 | 50 Hz |
| 高层策略 | Decimation 32 | 6.25 Hz |
| 低层回合时长 | 10 秒 | 500 个策略步 |
| 高层回合时长 | 24 秒 | 150 个策略步 |

每个高层动作会连续执行 8 个低层策略步。

## 项目结构

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

## 环境要求

当前移植版本的开发环境包括：

- Ubuntu Linux
- NVIDIA GPU 及兼容的显卡驱动
- NVIDIA Isaac Sim
- NVIDIA Isaac Lab 2.3.2
- Isaac Lab 配置的 Python 环境
- PyTorch
- SKRL

安装本项目前，请先完成 Isaac Sim 和 Isaac Lab 的安装并确认其可以正常运行。
官方安装方法请参考
[Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)。

## 安装方法

克隆本仓库：

```bash
git clone https://github.com/forerunner2/visual_wholebody_isaaclab.git
cd visual_wholebody_isaaclab
```

激活 Isaac Lab 使用的 Python 环境，然后安装本项目扩展：

```bash
python -m pip install -e source/visual_wholebody_isaaclab
python -m pip install skrl
```

检查环境是否成功注册：

```bash
python scripts/list_envs.py
```

## 已注册任务

当前仓库主要注册了以下任务：

```text
Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0
Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0
Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0
```

可以运行 `scripts/list_envs.py` 查看当前版本实际注册的任务名称。

## 训练流程

### 1. 训练低层策略

低层控制器使用从原项目迁移的自定义 RSL-RL 实现。该实现保留了本体感知、
特权信息和历史观测输入，以及腿部和机械臂双控制头、特权信息编码器、
状态历史编码器和内部 DAgger 风格的历史编码更新流程。

```bash
python scripts/rsl_rl/train_v1.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --num_envs 4096 --headless
```

运行训练完成的策略：

```bash
python scripts/rsl_rl/play_v1.py --task Isaac-VisualWholeBody-B1Z1-LowLevel-Direct-v0 --checkpoint /path/to/model.pt
```

> `scripts/rsl_rl/train.py` 和标准 Isaac Lab RSL-RL 配置仅作为参考保留。
> 标准网络产生的模型与自定义 `train_v1.py` 使用的网络结构不兼容。

### 2. 配置低层模型

训练高层教师策略前，需要在以下文件中设置 `low_policy_path`：

```text
source/visual_wholebody_isaaclab/visual_wholebody_isaaclab/
tasks/direct/pick_multi/pick_multi_env_cfg.py
```

如果对应脚本支持，也可以通过命令行参数传入低层模型路径。

### 3. 训练基于状态的高层教师策略

```bash
python scripts/skrl/train.py --task Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0 --num_envs 4096 --headless
```

教师策略接收仿真中的特权状态信息，并输出 9 维高层动作，包括：

- 末端执行器位置增量
- 末端执行器姿态指令
- 夹爪指令
- 移动底盘指令

### 4. 训练视觉学生策略

获得低层策略和高层教师策略模型后，使用 DAgger 训练视觉学生策略：

```bash
python scripts/dagger/train_student.py --task Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0 --num_envs 64 --low_policy /path/to/low_level_model.pt --teacher_ckpt /path/to/teacher_checkpoint.pt
```

视觉学生策略使用前置相机和腕部相机提供的深度与语义信息，
学习模仿基于状态的教师策略。

## 注意事项

- 大规模并行环境需要较多显存。如果出现显存不足，请减小 `--num_envs`。
- 原始 Isaac Gym 项目训练的模型不一定能直接用于本 Isaac Lab 版本。
- 模型是否兼容取决于训练脚本使用的观测结构和网络结构。
- 本项目仍在开发中，暂不保证完全复现原论文中的训练性能。
- 不建议将训练日志、模型检查点、缓存文件和实验输出提交到 Git 仓库。

## 原始项目与相关工作

本仓库移植自：

- 原始代码仓库：
  [Ericonaldo/visual_whole_body](https://github.com/Ericonaldo/visual_whole_body)
- 项目主页：
  [Visual Whole-Body Control](https://wholebody-b1.github.io/)
- 论文：
  [Visual Whole-Body Control for Legged Loco-Manipulation](https://arxiv.org/abs/2403.16967)

原始项目还使用或参考了以下项目：

- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [SKRL](https://github.com/Toni-SM/skrl)
- [DeepWBC](https://github.com/MarkFzp/Deep-Whole-Body-Control)

## 致谢

本项目的主要研究工作、算法设计和原始 Isaac Gym 实现均由
`visual_whole_body` 项目的原作者与贡献者完成。

本仓库的主要工作是将原项目适配到 Isaac Lab 框架。原始方法和研究成果的
相关贡献与荣誉均属于原作者。

## 引用

如果本项目对你的研究有所帮助，请引用原始论文：

```bibtex
@article{liu2024visual,
    title   = {Visual Whole-Body Control for Legged Loco-Manipulation},
    author  = {Liu, Minghuan and Chen, Zixuan and Cheng, Xuxin and
               Ji, Yandong and Yang, Ruihan and Wang, Xiaolong},
    journal = {arXiv preprint arXiv:2403.16967},
    year    = {2024}
}
```

在合适的情况下，也可以附上本 Isaac Lab 移植仓库的链接：

```text
https://github.com/forerunner2/visual_wholebody_isaaclab
```

## 许可证

本仓库包含从上游项目适配的代码和资源。使用或重新发布代码、模型及资源前，
请分别确认原始项目和各项第三方依赖的许可证要求。
