# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter that presents an Isaac Lab ``DirectRLEnv`` through the v1-style rsl_rl ``VecEnv`` interface.

The original ``visual_wholebody`` low-level stack (ported under
``visual_wholebody_isaaclab.learning.rsl_rl``) drives a plain-tensor environment:

    - ``get_observations()`` -> ``Tensor (N, num_obs)``
    - ``get_privileged_observations()`` -> ``None``
    - ``step(actions)`` -> ``(obs, priv, rew, arm_rew, dones, infos)``
    - ``episode_length_buf`` (get/set), ``max_episode_length``
    - ``num_envs``, ``num_obs``, ``num_actions``, ``num_privileged_obs``
    - ``cfg.env.num_proprio``, ``cfg.env.num_priv``, ``cfg.env.history_len``
    - ``p_gains``, ``d_gains``, ``default_dof_pos``

This wrapper translates between that interface and Isaac Lab's ``DirectRLEnv``,
including the original double reward channel (leg reward in ``rew``, arm reward
in ``arm_rew``) that the custom PPO expects.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.envs import DirectRLEnv


class VisualWholeBodyVecEnvWrapper:
    """v1-style VecEnv adapter for the B1+Z1 low-level environment."""

    def __init__(self, env: DirectRLEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length

        # observation / action dims (744 = 66 prop + 18 priv + 660 hist)
        self.num_obs = env.cfg.observation_space
        self.num_actions = env.cfg.action_space
        self.num_privileged_obs = None

        # expose the config fields the v1 runner reads through ``cfg.env``
        cfg = env.cfg
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(
                num_proprio=cfg.num_proprio,
                num_priv=cfg.num_priv,
                history_len=cfg.history_len,
            )
        )

    # ------------------------------------------------------------------ MDP

    def reset(self):
        obs_dict, _ = self.env.reset()
        obs = self._extract_obs(obs_dict)
        return obs, {}

    def get_observations(self) -> torch.Tensor:
        obs_dict = self.env._get_observations()
        return self._extract_obs(obs_dict)

    def get_privileged_observations(self):
        return None

    def step(self, actions: torch.Tensor):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # time-out bootstrapping for the v1 PPO
        if not self.env.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        obs = self._extract_obs(obs_dict)
        return obs, None, rew, self.arm_rew_buf, dones, extras

    def close(self):
        return self.env.close()

    # ------------------------------------------------------------------ attrs

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value

    @property
    def arm_rew_buf(self) -> torch.Tensor:
        """Arm reward channel (exposed by the low-level env as ``arm_rew_buf``)."""
        return self.env.arm_rew_buf

    @property
    def p_gains(self) -> torch.Tensor:
        return self.env.p_gains

    @property
    def d_gains(self) -> torch.Tensor:
        return self.env.d_gains

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return self.env.default_dof_pos

    def __str__(self):
        return f"<VisualWholeBodyVecEnvWrapper>"

    def __repr__(self):
        return str(self)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _extract_obs(obs_dict) -> torch.Tensor:
        """Pull the ``policy`` observation tensor out of the env observation dict."""
        return obs_dict["policy"]
