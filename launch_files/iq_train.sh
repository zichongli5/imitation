#!/bin/sh

# Set working directory to iq_learn
cd /home/zli911/imitation/baselines/IQ-Learn/iq_learn

# Hopper-v2
python train_iq.py env=cheetah agent=sac expert.demos=1 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0
