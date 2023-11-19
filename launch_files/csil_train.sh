#!/bin/sh

python train_csil.py --test_zoo True --env_id $1 --algo $2 --seed $3 
# --expert_episodes $4 --bc_epochs $5 --bc_batch_size $6 --bc_hidden_size $7 --bc_ent_weight $8 --bc_l2_weight $9
