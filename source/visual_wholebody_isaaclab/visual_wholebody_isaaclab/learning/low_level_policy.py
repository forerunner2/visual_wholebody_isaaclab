# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to load and run the frozen low-level policy inside the high-level environment.

The low-level policy is trained with the original custom RSL-RL stack (ported under
``visual_wholebody_isaaclab.learning.rsl_rl``). This wrapper loads the checkpoint
(``model.pt``) and exposes a flat-observation inference interface that supports the
original ``hist_encoding`` flag:

    - ``hist_encoding=False``: the actor uses the ``priv_encoder`` (privileged latent).
    - ``hist_encoding=True``:  the actor uses the ``StateHistoryEncoder`` over the
      observation history (deployment mode, no privileged information needed).

The observation is the 744-dim layout ``[prop(66), priv(18), hist(660)]`` produced by
the low-level environment. When ``hist_encoding=True`` the network only reads the
``prop`` and ``hist`` slices, so the high-level environment may feed a ``[prop, hist]``
tensor of 726 dims.
"""

from __future__ import annotations

import os

import torch

from visual_wholebody_isaaclab.learning.rsl_rl.actor_critic import ActorCritic


class LowLevelPolicy:
    """Frozen low-level B1+Z1 policy with a flat-obs inference interface."""

    def __init__(
        self,
        checkpoint_path: str,
        obs_space: int,
        action_space: int,
        device: str,
        num_prop: int = 66,
        num_priv: int = 18,
        num_hist: int = 10,
        num_leg_actions: int = 12,
        num_arm_actions: int = 6,
        actor_hidden_dims: tuple | list = (128,),
        critic_hidden_dims: tuple | list = (128,),
        leg_control_head_hidden_dims: tuple | list = (128, 128),
        arm_control_head_hidden_dims: tuple | list = (128, 128),
        priv_encoder_dims: tuple | list = (64, 20),
        activation: str = "elu",
    ):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Low-level policy checkpoint not found: {checkpoint_path}")
        self.device = device
        self.num_prop = num_prop
        self.num_priv = num_priv
        self.num_hist = num_hist

        # build the custom ActorCritic (same structure as during training)
        self.model = ActorCritic(
            num_prop,
            num_prop,
            action_space,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            leg_control_head_hidden_dims=leg_control_head_hidden_dims,
            arm_control_head_hidden_dims=arm_control_head_hidden_dims,
            priv_encoder_dims=priv_encoder_dims,
            activation=activation,
            num_leg_actions=num_leg_actions,
            num_arm_actions=num_arm_actions,
            adaptive_arm_gains=False,
            adaptive_arm_gains_scale=10.0,
            output_tanh=False,
            num_priv=num_priv,
            num_hist=num_hist,
            num_prop=num_prop,
            init_std=[[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        print(f"[LowLevelPolicy] Loaded checkpoint from {checkpoint_path}")

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, hist_encoding: bool = True) -> torch.Tensor:
        """Run inference on a flat observation tensor.

        Args:
            obs: Flat observation of shape ``(N, obs_space)``. ``obs_space`` is 744
                (prop+priv+hist) or 726 (prop+hist) when ``hist_encoding=True``.
            hist_encoding: Use the history encoder instead of the priv encoder.

        Returns:
            Actions of shape ``(N, action_space)``.
        """
        obs = obs.to(self.device)
        return self.model.act_inference(obs, hist_encoding=hist_encoding)
