# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train the visual student with DAgger (teacher-student) for the B1+Z1
pick-multi task.

The state teacher is trained first with ``scripts/skrl/train.py --task
Isaac-VisualWholeBody-B1Z1-PickMulti-Teacher-Direct-v0``. Its checkpoint is then
loaded here to label the student's image observations.

Example:
    python scripts/dagger/train_student.py \
        --task Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0 \
        --num_envs 64 \
        --teacher_ckpt <path-to-teacher>.pt \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train the DAgger visual student.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-VisualWholeBody-B1Z1-PickMulti-Vision-Direct-v0")
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Path to the trained state-teacher checkpoint.")
parser.add_argument("--low_policy", type=str, required=True, help="Path to the trained low-level policy checkpoint.")
parser.add_argument("--headless", action="store_true", help="Run headless.")
parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps to train.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # DAgger student needs cameras

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import visual_wholebody_isaaclab.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

from visual_wholebody_isaaclab.learning.dagger import DAgger, DAGGER_DEFAULT_CONFIG
from visual_wholebody_isaaclab.learning.dagger.dagger_rnn import DAgger_RNN, DAGGER_DEFAULT_CONFIG as DAGGER_RNN_CONFIG
from visual_wholebody_isaaclab.learning.dagger.dagger_trainer import DAggerTrainer
from visual_wholebody_isaaclab.learning.dagger.feature_extractor import DepthOnlyFCBackbone54x96

# camera mode "full": 12 channels (4 channels x 3 history)
CAMERA_MODE = "full"


class VisualWholeBodyDaggerWrapper:
    """Pass-through wrapper exposing the two-channel observation dict.

    The stock ``SkrlVecEnvWrapper`` collapses the env's observation dict to a single
    flat tensor and drops the ``states``/``obs`` channels that the DAgger trainer needs
    (see ``learning/dagger/dagger_trainer.py``). This wrapper forwards the raw env's
    observation dict straight through, mirroring the original ``IsaacGymPreview3Wrapper``.
    """

    def __init__(self, env):
        self._env = env
        self.device = env.device if isinstance(env.device, torch.device) else torch.device(env.device)
        self.num_agents = 1

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def max_episode_length(self):
        env = self._env
        if hasattr(env, "max_episode_length"):
            return env.max_episode_length
        try:
            return int(env.max_episode_length_s / env.step_dt)
        except Exception:
            return 150

    def reset(self):
        obs, info = self._env.reset()
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self._env.step(actions)
        return obs, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        self._env.close()


class Policy(DeterministicMixin, Model):
    """Visual student policy: image backbone + GRU + MLP head."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs=1,
        num_layers=1,
        hidden_size=128,
        sequence_length=16,
        mode="full",
        use_roboinfo=True,
        use_gru=True,
        num_channel=12,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.mode = mode
        self.use_roboinfo = use_roboinfo
        self.use_gru = use_gru

        # proprioception tail appended to the image features (see the original model)
        input_size = 28 + 9 + 64
        if use_roboinfo:
            input_size += 24

        if use_gru:
            self.gru = nn.GRU(
                input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, self.hidden_size), nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh(),
            )

        self.depth_extractor = DepthOnlyFCBackbone54x96(
            latent_dim=64, output_activation=None, num_channel=num_channel
        )
        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64), nn.ReLU(), nn.Linear(64, self.num_actions))

        if use_gru:
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [(self.num_layers, self.num_envs, self.hidden_size)],
            }
        }

    def compute(self, inputs, role):
        states_raw = inputs["states"]
        img_dim = 61 if self.use_roboinfo else 37  # robot obs tail (see pick_multi_env)
        images = states_raw[:, :-img_dim]
        images = images.reshape(-1, 12, 54, 96)

        depth_feature = self.depth_extractor(images)
        states = torch.cat([states_raw[:, -img_dim:], depth_feature], dim=1)

        hidden_states = None
        if self.use_gru:
            hidden_states = inputs["rnn"][0]
            if self.training:
                rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
                hidden_states = hidden_states.view(
                    self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
                )[:, :, 0, :].contiguous()
                terminated = inputs.get("terminated", None)
                if terminated is not None and torch.any(terminated):
                    rnn_outputs = []
                    terminated = terminated.view(-1, self.sequence_length)
                    indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [
                        self.sequence_length
                    ]
                    for i in range(len(indexes) - 1):
                        i0, i1 = indexes[i], indexes[i + 1]
                        rnn_output, hidden_states = self.gru(rnn_input[:, i0:i1, :], hidden_states)
                        hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                        rnn_outputs.append(rnn_output)
                    rnn_output = torch.cat(rnn_outputs, dim=1)
                else:
                    rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
            else:
                rnn_input = states.view(-1, 1, states.shape[-1])
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)

            prev_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
            actions = self.net(prev_output)
            return actions, {"rnn": [hidden_states]}
        else:
            prev_output = self.mlp(states)
            actions = self.net(prev_output)
            return actions, {}


def main():
    set_seed(args_cli.seed)

    # load env config and override
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.low_policy_path = args_cli.low_policy
    env_cfg.scene.num_envs = min(env_cfg.scene.num_envs, 400)  # keep camera memory in check

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = VisualWholeBodyDaggerWrapper(env)

    device = env.device
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

    # student = images + 61-dim proprio tail (state_space); teacher = 1094-dim state obs.
    student_state_space = env.state_space
    teacher_obs_space = env.observation_space
    student_action_space = env.action_space

    model_dagger = {
        "policy": Policy(
            student_state_space,
            student_action_space,
            device,
            num_envs=env.num_envs,
            mode=CAMERA_MODE,
            use_roboinfo=True,
            use_gru=True,
            num_channel=12,
        )
    }

    cfg_dagger = DAGGER_RNN_CONFIG.copy()
    cfg_dagger["rollouts"] = 24
    cfg_dagger["learning_epochs"] = 5
    cfg_dagger["mini_batches"] = 3
    cfg_dagger["learning_rate"] = 5e-5
    cfg_dagger["grad_norm_clip"] = 1.0
    cfg_dagger["experiment"]["directory"] = os.path.join("logs", "dagger", "b1z1_pickmulti_student")
    cfg_dagger["experiment"]["checkpoint_interval"] = 1000

    agent = DAgger_RNN(
        models=model_dagger,
        memory=memory,
        cfg=cfg_dagger,
        observation_space=teacher_obs_space,
        action_space=student_action_space,
        state_space=student_state_space,
        device=device,
    )

    # load the trained state-teacher (skrl PPO) as the labeler
    teacher = load_teacher_agent(env, device, args_cli.teacher_ckpt)

    cfg_trainer = {"timesteps": args_cli.timesteps, "teacher_pretrain": True, "pretrain_timesteps": 4000}
    trainer = DAggerTrainer(cfg=cfg_trainer, env=env, agents=agent, teacher_agents=teacher)
    trainer.train()


def load_teacher_agent(env, device, checkpoint_path):
    """Instantiate the skrl PPO teacher agent and load its checkpoint.

    NOTE: the teacher architecture is defined by the SKRL ``PPO`` agent with the
    teacher observation space. For simplicity we reuse the teacher's own skrl
    config (see scripts/skrl/train.py) via the model instantiator.
    """
    from skrl.agents.torch.ppo import PPO as SkrlPPO
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    teacher_obs_space = env.observation_space
    teacher_action_space = env.action_space

    memory_teacher = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

    cfg_ppo = {
        "rollouts": 24,
        "learning_epochs": 5,
        "mini_batches": 6,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 5e-4,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": teacher_obs_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {"directory": "b1z1_pickmulti_teacher", "experiment_name": ""},
    }

    # Build the teacher models with the same architecture used for training
    # (a two-hidden-layer MLP policy / value over the 1094-dim teacher observation).
    from skrl.models.torch import GaussianMixin

    class TeacherPolicy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device):
            GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True)
            Model.__init__(self, observation_space, action_space, device)
            self.net = nn.Sequential(
                nn.Linear(self.num_observations, 512), nn.ELU(),
                nn.Linear(512, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, self.num_actions),
            )

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    class TeacherValue(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device):
            DeterministicMixin.__init__(self, clip_actions=False)
            Model.__init__(self, observation_space, action_space, device)
            self.net = nn.Sequential(
                nn.Linear(self.num_observations, 512), nn.ELU(),
                nn.Linear(512, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 1),
            )

        def compute(self, inputs, role):
            return self.net(inputs["states"]), {}

    ppo_models = {
        "policy": TeacherPolicy(teacher_obs_space, teacher_action_space, device),
        "value": TeacherValue(teacher_obs_space, teacher_action_space, device),
    }
    teacher = SkrlPPO(
        models=ppo_models,
        memory=memory_teacher,
        observation_space=teacher_obs_space,
        action_space=teacher_action_space,
        cfg=cfg_ppo,
        device=device,
    )
    teacher.load(checkpoint_path)
    teacher.set_running_mode("eval")
    return teacher


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        simulation_app.close()
