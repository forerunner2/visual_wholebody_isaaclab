# Visual Whole-Body（Isaac Lab）

本项目是基于原始 [visual_wholebody](https://github.com/Ericonaldo/visual_wholebody)
项目移植并适配到 **Isaac Lab 2.3.2** 的版本。

原项目主要实现了基于 **Unitree B1 四足机器人 + Z1 机械臂** 的全身运动与操作（Whole-Body Loco-Manipulation）。
本项目在原始实现的基础上，将相关任务环境、训练流程以及视觉学习模块适配到 Isaac Lab 2.3.2。

> **项目来源说明：**
> 本项目包含来自原始 `visual_wholebody` 项目的移植和改编代码。
> 使用、修改和再发布本项目时，请同时遵守原项目的许可证要求。

---

## 项目架构

整个项目主要包含三个层级：

```text
低层环境（Low-Level）
    DirectRLEnv + RSL-RL
    输入：基座速度指令 + 末端执行器目标位姿
    输出：B1 腿部强化学习动作 + Z1 差分逆运动学目标

        ↓ 冻结低层策略

Pick-Multi 状态教师环境（Teacher）
    DirectRLEnv + SKRL PPO，无相机
    输入：物体 / 桌面 / 机器人真实状态
    输出：末端位姿增量 + 夹爪动作 + 基座指令（9维）
    调用冻结的低层策略

        ↓ DAgger

Pick-Multi 视觉学生环境（Vision Student）
    DirectRLEnv + TiledCamera
    输入：前置相机 + 腕部相机的深度图和语义分割图像
    输出：与教师策略相同的9维动作
    使用 DAgger + GRU 训练视觉学生策略
