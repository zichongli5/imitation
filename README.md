# Requirements

To create your virtual environment, you can run the following commands


```
conda create -n IL python=3.8
conda activate IL
pip3 install torch torchvision torchaudio
pip install stable-baselines3[extra]
pip install gymnasium
pip install tensorboard
pip install rl_zoo3
pip install pybullet
pip install pybullet_envs_gymnasium
# for IQ-learn
pip install wandb
pip install hydra-core --upgrade
pip install tensorboardX
pip install termcolor
<!-- pip install 'shimmy>=0.2.1' -->
```

# Example

To run our method, you should first train an IQ-learn model by
```
cd baselines/IQ-Learn/iq_learn
python train_iq.py env=cheetah agent=sac expert.demos=1 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0
```
For other environment, scripts can be found in 'baselines/IQ-Learn/iq_learn/scripts'; However, for mujoco tasks, iq-learn use the original mujoco env but we use pybullet env, so you may need to alter the config file in 'baselines/IQ-Learn/iq_learn/conf/env' and change the env name to pybullet version.

# Tuning
