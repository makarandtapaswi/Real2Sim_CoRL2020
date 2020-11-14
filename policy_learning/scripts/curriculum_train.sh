#!/bin/bash
# hand position randomization x hand orientation randomization

for seed in 0 1 2 3 4
do

  # fixed x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_null_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 1e-5 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_null_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 1e-5 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_null_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 1e-5 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_null_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 1e-5 --fixed_hpos_std --without_object_obs -seed $seed

  # dr x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.25 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.25 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.25 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.25 --fixed_hpos_std --without_object_obs -seed $seed

  # adr x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 --without_object_obs -seed $seed

  # adr x adr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_adr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -angle_bound_scale 0.01 --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_adr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -angle_bound_scale 0.01 --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_adr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -angle_bound_scale 0.01 --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_adr_adr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -angle_bound_scale 0.01 --without_object_obs -seed $seed

  # dr smaller x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr01_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.01 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr01_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.01 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr01_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.01 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr01_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.01 --fixed_hpos_std --without_object_obs -seed $seed

   dr smaller x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr05_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.05 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr05_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.05 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr05_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.05 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr05_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.05 --fixed_hpos_std --without_object_obs -seed $seed

  # dr smaller x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr10_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.10 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr10_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.10 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr10_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.10 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr10_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.10 --fixed_hpos_std --without_object_obs -seed $seed

  # dr smaller x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr15_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.15 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr15_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.15 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr15_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.15 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr15_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.15 --fixed_hpos_std --without_object_obs -seed $seed

  # dr smaller x dr
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr20_dr_gt_true_act_86 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.20 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr20_dr_gt_true_act_87 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.20 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr20_dr_gt_true_act_93 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.20 --fixed_hpos_std --without_object_obs -seed $seed
  python policy_learning/manipulation_learning.py -log_dir data/policies  -name curriculum_dr20_dr_gt_true_act_94 --linear_lr -states_folder states/1sA -cuda_id 0 -hpos_std 0.20 --fixed_hpos_std --without_object_obs -seed $seed

done