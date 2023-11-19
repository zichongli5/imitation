#!/bin/sh

python train_bc.py --test_zoo True --env_id $1 --algo $2 --seed $3 --expert_episodes $4 --bc_algo $5 --bc_epochs $6 --bc_batch_size $7 --bc_hidden_size $8 --bc_ent_weight $9 --bc_l2_weight ${10}

