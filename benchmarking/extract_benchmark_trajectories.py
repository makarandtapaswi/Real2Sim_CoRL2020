#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-05-8
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pyphysx_utils.transformations import multiply_transformations
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt_utils import args
from rlpyt_utils.agents_nn import AgentPgDiscrete, AgentPgContinuous

from policy_learning.agents import ModelPgNNContinuousSelective
from simulation.utils.rotations import *
from simulation.envs import GripperCylinderEnv
from simulation.utils.states import ActionClasses
from tqdm import tqdm
import quaternion as npq

parser = args.get_default_rl_parser()
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('benchmark_file', type=str)
parser.add_argument('output_dir')
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--realtime', dest='realtime', action='store_true')
parser.add_argument('--without_object_obs', dest='without_object_obs', action='store_true')
args.add_default_ppo_args(parser)
options = parser.parse_args()

options.greedy_eval = True

print('Using options:', options)

output_dir = Path(options.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
bench_data = pd.read_csv(options.benchmark_file)

""" Get action class from policy name. """
action_class = None
if 'vid' in options.name:
    video_id = int(options.name.split('_')[-1])
    for action in ActionClasses:
        if action.contains_video(video_id):
            action_class = action
            break
elif 'act' in options.name:
    action_id = int(options.name.split('_')[-1])
    action_class = ActionClasses(action_id)
else:
    raise NotImplementedError('Name must specify what to learn. Use vid_{id} or act_{id} at the end of your name')


def benchmark_sample(env):
    if not hasattr(benchmark_sample, "bid"):
        benchmark_sample.bid = 0
    d0 = bench_data.loc[min(benchmark_sample.bid, bench_data.shape[0] - 1)]
    q = quat_from_azimuth_elevation(d0['hazi'], d0['hele'])
    tip_pos = np.array([d0['hposx'], d0['hposy'], d0['hposz']])
    obj_pos = np.array([d0['o0posx'], d0['o0posy'], d0['o0posz']])
    if action_class.is_single_object():
        return (tip_pos, q), GripperCylinderEnv.FINGER_OPEN_POS, (obj_pos, npq.one), d0['o0hei'], d0['o0rad']
    o1pose = multiply_transformations((tip_pos, q), (np.zeros(3), quat_from_euler('y', np.pi / 2)))
    return (tip_pos, q), d0['o1rad'], (obj_pos, npq.one), d0['o0hei'], d0['o0rad'], o1pose, d0['o1hei'], d0['o1rad']


horizon = 120

sampler_cls = SerialSampler
sampler = sampler_cls(
    EnvCls=GripperCylinderEnv,
    env_kwargs=dict(
        render=options.render, realtime=options.realtime,
        reward_params=GripperCylinderEnv.get_default_reward_params(),
        reset_state_sampler=benchmark_sample,
        horizon=horizon,
        reset_on_singularity=False,
        reset_on_plane_hit=False,
        two_objects=not action_class.is_single_object(),
        action_start_time=3.,
        action_end_time=7.,
        open_gripper_on_leave=action_class != ActionClasses.PICK_UP,
        close_gripper_on_leave=action_class == ActionClasses.PICK_UP,
    ),
    batch_T=horizon, batch_B=1, max_decorrelation_steps=0
)

algo = args.get_ppo_from_options(options)
agent = AgentPgContinuous(
    options.greedy_eval,
    ModelCls=ModelPgNNContinuousSelective,
    initial_model_state_dict=args.load_initial_model_state(options),
    model_kwargs=dict(
        policy_hidden_sizes=[128, 128, 128], policy_hidden_nonlinearity=torch.nn.Tanh,
        value_hidden_sizes=[128, 128, 128], value_hidden_nonlinearity=torch.nn.Tanh,
        policy_inputs_indices=list(range(8)) if options.without_object_obs else None,
    )
)

runner = MinibatchRl(
    algo=algo, agent=agent, sampler=sampler, log_traj_window=1, seed=options.seed, n_steps=1,
    log_interval_steps=int(1 * horizon), affinity=args.get_affinity(options)
)
runner.startup()
for i in tqdm(range(bench_data.shape[0])):
    benchmark_sample.bid = i + 1
    sampler.obtain_samples(i)
    GripperCylinderEnv.df_from_observations(sampler.samples_np.env.observation[:, 0, :]).to_csv(
        '{}/trajectory_{}.csv'.format(output_dir, benchmark_sample.bid - 1))
