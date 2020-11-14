#!/bin/bash

for seed in 0 1 2 3 4 5 6 7 8 9
do
  for action in 86 87 93 94
  do
    for set in 1sA 1sC
    do
      python policy_learning/manipulation_learning.py -log_dir data/policies -name baseline_max_${set}_act_${action} --linear_lr -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs --max_reward -seed $seed
    done
  done

  for action in 104 105 107 112
  do
    for set in 2sA 2sC
    do
      python policy_learning/manipulation_learning.py -log_dir data/policies -name baseline_max_${set}_act_${action} --linear_lr -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs --max_reward -seed $seed
    done
  done

  for action in 47
  do
    for set in 1sA 1sC
    do
      python policy_learning/manipulation_learning.py -log_dir data/policies -name baseline_max_${set}_act_${action} --linear_lr -states_folder states/${set} -angle_bound_scale 0.01 --without_object_obs --max_reward -dz 0 -seed $seed
    done
  done

done