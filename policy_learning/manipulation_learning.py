#!/usr/bin/env python

# Copyright (c) CTU -- All Rights Reserved
# Created on: 2020-06-5
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>

import time
import torch
from pathlib import Path
from multiprocessing import Manager

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging.logger import record_tabular, record_tabular_misc_stat

from rlpyt_utils import args
from rlpyt_utils.runners.minibatch_rl import MinibatchRlWithLog
from rlpyt_utils.agents_nn import AgentPgContinuous

from simulation.envs import GripperCylinderEnv
from simulation.utils.states import StateTrajectories, ActionClasses, StateTrajectory
from simulation.utils.rotations import *
from policy_learning.agents import ModelPgNNContinuousSelective

parser = args.get_default_rl_parser()
args.add_default_ppo_args(parser, clip_grad_norm=100, discount=1.)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-states_folder', type=str, default='states/all')

# Evaluation args
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--realtime', dest='realtime', action='store_true')
parser.add_argument('--store_video', dest='store_video', action='store_true')
parser.add_argument('--store_fig', dest='store_fig', action='store_true')
parser.add_argument('--noshow', dest='show', action='store_false')
parser.add_argument('-eval_iterations', type=int, default=1)
parser.add_argument('-hpos_std', type=float, default=0.005, help='Specify start STD for HPOS randomization.')
parser.add_argument('--fixed_hpos_std', dest='fixed_hpos_std', action='store_true')
parser.add_argument('-angle_bound_scale', type=float, default=1.)
parser.add_argument('--without_object_obs', dest='without_object_obs', action='store_true')
parser.add_argument('--norot', dest='norot', action='store_true')
parser.add_argument('-dz', type=float, default=0.2)
parser.add_argument('--max_reward', dest='max_reward', action='store_true', help='Use maximum reward as in DeepMimic instead of default average reward computation.')
parser.add_argument('--add_obstacle', dest='add_obstacle', action='store_true')
options = parser.parse_args()

video_id = None
action_id = None

if 'vid' in options.name:
    video_id = int(options.name.split('_')[-1])
elif 'act' in options.name:
    action_id = int(options.name.split('_')[-1])
else:
    raise NotImplementedError('Name must specify what to learn. Use vid_{id} or act_{id} at the end of your name')

horizon = 120
if horizon == 12 * 12:
    action_start_time, action_end_time = 4., 8.
elif horizon == 120:
    action_start_time, action_end_time = 3., 7.
elif horizon == 60:
    action_start_time, action_end_time = 2., 4.
else:
    raise NotImplementedError('')

state_trajectories = StateTrajectories(folder=options.states_folder)
if options.norot:
    for s in state_trajectories.trajectories:
        s.df['o0rot0'] = 0
        s.df['o0rot1'] = 0
        s.df['o0rot2'] = 0
        s.df['o1rot0'] = 0
        s.df['o1rot1'] = 0
        s.df['o1rot2'] = 0

for s in state_trajectories.trajectories:
    s.transform_to_object_frame(xyz_scale=1.5)
    s.set_start_and_final_time(action_start_time, action_end_time)

if video_id is not None:
    demo_trajectories = [state_trajectories.get_from_video_id(video_id)]
elif action_id is not None:
    demo_trajectories = state_trajectories.get_all_for_action(ActionClasses(action_id))
else:
    raise NotImplementedError('')

eval_dir = Path('/tmp/sth_single_object/')
eval_dir.mkdir(exist_ok=True, parents=True)
eval_id = '{}_{}'.format(time.strftime("%Y%m%d_%H%M%S"), video_id or action_id)
video_filename = '{}/{}_render.mp4'.format(eval_dir, eval_id) if options.store_video else None
batch_mul_per_iter = 1
sampler_cls = SerialSampler if args.is_evaluation(options) else CpuSampler if options.cuda_id is None else GpuSampler

manager = Manager()
shared_sampler_dict = manager.dict()
shared_sampler_dict['hpos_std'] = np.clip(options.hpos_std, 1e-5, 0.25)
shared_sampler_dict['angle_bound_scale'] = options.angle_bound_scale

sampler = sampler_cls(
    EnvCls=GripperCylinderEnv,
    env_kwargs=dict(
        render=options.render,
        realtime=options.realtime,
        video_filename=video_filename,
        reward_params=GripperCylinderEnv.get_default_reward_params(
            demo_obj_pos=0.5, demo_hand_pos=0.2, demo_hand_azi_ele=0.2,
            demo_obj_rot=0. if options.norot else 0.05,
            demo_touch=0.05,
        ),
        reset_state_sampler=GripperCylinderEnv.randomized_reset_state_sampler(
            shared_sampler_dict=shared_sampler_dict, orad_min=0.04, orad_max=0.05, ohei_min=0.04, ohei_max=0.1,
        ),
        horizon=horizon,
        state_trajectories=demo_trajectories,
        reset_on_singularity=not args.is_evaluation(options),
        reset_on_plane_hit=not args.is_evaluation(options),
        action_start_time=action_start_time,
        action_end_time=action_end_time,
        two_objects=not demo_trajectories[0].get_action().is_single_object(),
        dz=options.dz,
        open_gripper_on_leave=demo_trajectories[0].get_action() != ActionClasses.PICK_UP,
        close_gripper_on_leave=demo_trajectories[0].get_action() == ActionClasses.PICK_UP,
        use_max_reward=options.max_reward,
        add_obstacle=options.add_obstacle,
    ),
    batch_T=horizon if args.is_evaluation(options) else horizon * 1 * batch_mul_per_iter,
    batch_B=1 if args.is_evaluation(options) else 70,
    max_decorrelation_steps=0
)

agent = AgentPgContinuous(
    options.greedy_eval,
    ModelCls=ModelPgNNContinuousSelective,
    initial_model_state_dict=args.load_initial_model_state(options),
    model_kwargs=dict(
        policy_hidden_sizes=[128, 128, 128], policy_hidden_nonlinearity=torch.nn.Tanh,
        value_hidden_sizes=[128, 128, 128], value_hidden_nonlinearity=torch.nn.Tanh,
        init_log_std=np.log(0.2),
        policy_inputs_indices=list(range(8)) if options.without_object_obs else None,
    ),
)


def log_diagnostics(itr, algo, agent, sampler):
    if itr > 0:
        shared_sampler_dict['angle_bound_scale'] = np.minimum(options.angle_bound_scale + 0.004 * (itr - 0), 1.)

    if itr > 500:
        if not options.fixed_hpos_std:
            shared_sampler_dict['hpos_std'] = np.minimum(options.hpos_std + 0.0005 * (itr - 500), 0.25)

    record_tabular('agent/hpos_std', shared_sampler_dict['hpos_std'])
    record_tabular('agent/angle_bound_scale', shared_sampler_dict['angle_bound_scale'])

    std = agent.model.log_std.exp().data.cpu().numpy()
    for i in range(std.shape[0]):
        record_tabular('agent/std{}'.format(i), std[i])

    record_tabular_misc_stat('final_obj_position_x', sampler.samples_np.env.observation[sampler.samples_np.env.done, 8])


runner = MinibatchRlWithLog(
    algo=args.get_ppo_from_options(options), agent=agent, sampler=sampler, log_traj_window=70 * batch_mul_per_iter,
    seed=options.seed,
    n_steps=int(1000 * batch_mul_per_iter * 70 * horizon),
    log_interval_steps=int(70 * batch_mul_per_iter * horizon * 100),
    affinity=args.get_affinity(options),
    log_diagnostics_fun=log_diagnostics
)

if not args.is_evaluation(options):
    with args.get_default_context(options):
        runner.train()
else:
    runner.startup()
    renderer = sampler.collector.envs[0].renderer
    while sampler.collector.envs[0].renderer.is_active:
        sampler.obtain_samples(0)
    runner.shutdown()
