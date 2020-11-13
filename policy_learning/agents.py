#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-07-8
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from rlpyt.agents.pg.gaussian import *
from rlpyt.agents.pg.categorical import *
from rlpyt.models.mlp import MlpModel
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class ModelPgNNContinuousSelective(torch.nn.Module):
    def __init__(self, observation_shape, action_size,
                 policy_hidden_sizes=None, policy_hidden_nonlinearity=torch.nn.Tanh,
                 value_hidden_sizes=None, value_hidden_nonlinearity=torch.nn.Tanh,
                 init_log_std=0., min_std=0.,
                 normalize_observation=False,
                 norm_obs_clip=10,
                 norm_obs_var_clip=1e-6,
                 policy_inputs_indices=None,
                 ):
        super().__init__()
        self.min_std = min_std
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        self.policy_inputs_indices = policy_inputs_indices if policy_inputs_indices is not None else list(
            range(input_size))

        policy_hidden_sizes = [400, 300] if policy_hidden_sizes is None else policy_hidden_sizes
        value_hidden_sizes = [400, 300] if value_hidden_sizes is None else value_hidden_sizes
        self.mu = MlpModel(input_size=len(self.policy_inputs_indices), hidden_sizes=policy_hidden_sizes,
                           output_size=action_size,
                           nonlinearity=policy_hidden_nonlinearity)
        self.v = MlpModel(input_size=input_size, hidden_sizes=value_hidden_sizes, output_size=1,
                          nonlinearity=value_hidden_nonlinearity, )
        self._log_std = torch.nn.Parameter((np.log(np.exp(init_log_std) - self.min_std)) * torch.ones(action_size))
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation

    @property
    def log_std(self):
        return (self._log_std.exp() + self.min_std).log()

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                                      obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)
        obs_flat = observation.view(T * B, -1)
        mu = self.mu(obs_flat[:, self.policy_inputs_indices])
        v = self.v(obs_flat).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)

        return mu, log_std, v

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)
