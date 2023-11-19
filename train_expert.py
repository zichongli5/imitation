import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from rl_zoo3 import ALGOS
from rl_zoo3.utils import get_wrapper_class
from imitation.data import rollout


import numpy as np
import os
import tqdm
import yaml
import argparse
from collections import defaultdict
import pickle

from utils import create_env, evaluate, get_saved_hyperparams, preprocess_hyperparams, load_pretrained_expert


def train_expert(args):
    '''
    Train expert agent
    '''
    env_id = args.env_id
    algo = args.algo
    config_path = args.config_path
    save_path = args.save_path
    seed = args.seed
    save_freq = args.save_freq
    eval_freq = args.eval_freq

    # Get the right path
    config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
    if args.use_sde:
        save_path = os.path.join(save_path, env_id, algo+'_sde')
    else:
        save_path = os.path.join(save_path, env_id, algo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load hyperparameters from yaml file
    hyperparams = get_saved_hyperparams(config_path)
    processed_hyperparams = preprocess_hyperparams(hyperparams)
    n_envs = hyperparams['n_envs']
    print(hyperparams)
    print(processed_hyperparams)

    # Set some special parameters
    n_timesteps = args.n_timesteps if args.n_timesteps > 0 else hyperparams['n_timesteps']
    if not args.use_sde and 'use_sde' in processed_hyperparams.keys():
        processed_hyperparams['use_sde'] = False

    # Create environment
    env_wrapper = get_wrapper_class(hyperparams)
    env = create_env(env_id, n_envs=n_envs, norm_obs=hyperparams['norm_obs'], norm_reward=hyperparams['norm_reward'], seed=seed, env_wrapper=env_wrapper)
    eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams['norm_obs'], norm_reward=False, seed=seed, env_wrapper=env_wrapper)

    # Create model
    model = ALGOS[algo](env=env, seed=seed, verbose=1, **processed_hyperparams)
    # batch_size, n_epochs, n_steps, gae_lambda, policy_kwargs
    print('Model created!')

    # Create Callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path, log_path=save_path, eval_freq=max(eval_freq // n_envs,1), n_eval_episodes = 10, deterministic=True, render=False)
    if save_freq > 0:
        checkpoint_callback = CheckpointCallback(save_freq=max(save_freq // n_envs, 1), save_path=save_path, name_prefix=algo+env_id, verbose=1)
        callback = CallbackList([checkpoint_callback, eval_callback])
    else:
        callback = eval_callback
    print('Setting Callback! Save model to {}'.format(save_path))

    # Train model
    model.learn(total_timesteps=n_timesteps, callback=callback)

    # Save stats
    env.save(os.path.join(save_path, env_id+'_stats.pkl'))
    print('Complete Training! Save stats to {}'.format(os.path.join(save_path, env_id+'_stats.pkl')))

    return 0

def test_expert(args):
    assert os.path.exists(args.weight_path), "Wrong path to the pretrained expert!"
    env_id = args.env_id
    algo = args.algo
    config_path = args.config_path
    seed = args.seed

    # Get the right path
    config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
    model_path = os.path.join(args.weight_path, env_id, algo, args.model_name)
    stats_path = os.path.join(args.weight_path, env_id, algo, env_id+'_stats.pkl')

    # Load hyperparameters from yaml file
    hyperparams = get_saved_hyperparams(config_path)

    # Create environment
    eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path)

    # Evaluate the trained agent
    expert = load_pretrained_expert(eval_env, algo, model_path=model_path, eval=True, eval_steps=100)

    return eval_env, expert

def test_zoo_expert(args):
    env_id = args.env_id
    algo = args.algo
    config_path = args.config_path
    seed = args.seed

    # Get the right path
    zoo_path = '/home/zli911/imitation/expert_files/rl-trained-agents'
    config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
    stats_path = os.path.join(zoo_path, algo, env_id+'_1', env_id, 'vecnormalize.pkl')
    model_path = os.path.join(zoo_path, algo, env_id+'_1',env_id+'.zip')

    # Load hyperparameters from yaml file
    hyperparams = get_saved_hyperparams(config_path)

    # Create environment
    print(hyperparams)
    env_wrapper = get_wrapper_class(hyperparams)
    eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path, env_wrapper=env_wrapper, manual_load=True)

    # Evaluate the trained agent
    expert = load_pretrained_expert(eval_env, algo, model_path=model_path, eval=True, eval_steps=10)

    return eval_env, expert

def save_expert_trajectory_IQ(args):
    if args.test_zoo:
        eval_env, expert = test_zoo_expert(args)
    else:
        eval_env, expert = test_expert(args)

    # Collect expert trajectories
    eval_env.seed(args.seed)
    rng=np.random.default_rng(args.seed)
    rollouts = rollout.rollout(expert, eval_env, 
                               rollout.make_sample_until(min_timesteps=None, min_episodes=50),
                               unwrap=False,
                               verbose=True,
                               rng=rng,
                               )
    # Truncate rollouts based on expert_episodes
    stats = rollout.rollout_stats(rollouts)
    print(f"Rollout stats: {stats}")
    # print(rollouts)

    expert_trajs = defaultdict(list)
    for traj_num in range(len(rollouts)):
        expert_trajs["states"].append(rollouts[traj_num].obs[:-1].reshape(len(rollouts[traj_num].acts),1,-1))
        expert_trajs["next_states"].append(rollouts[traj_num].obs[1:].reshape(len(rollouts[traj_num].acts),1,-1))
        expert_trajs["actions"].append(rollouts[traj_num].acts.reshape(len(rollouts[traj_num].acts),1,-1))
        expert_trajs["rewards"].append(rollouts[traj_num].rews.reshape(len(rollouts[traj_num].acts),1))
        expert_trajs["dones"].append(np.array([[False]]*len(rollouts[traj_num].acts)))
        expert_trajs["lengths"].append(len(rollouts[traj_num].acts))

    save_path = os.path.join(args.save_path, args.env_id, args.algo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'{save_path}/IQ_trajs_norm.pkl', 'wb') as f:
        pickle.dump(expert_trajs, f)
    print('IQ-learn expert saved!')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--algo", default='ppo', type=str)
    parser.add_argument("--config_path", default='/home/zli911/imitation/expert_files/rl-trained-agents/', type=str)
    parser.add_argument("--save_path", default='/home/zli911/imitation/expert_files/', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_freq", default=-1, type=int)
    parser.add_argument("--eval_freq", default=100000, type=int)
    parser.add_argument("--n_timesteps", default=-1, type=int)
    parser.add_argument("--use_sde", default=False, type=bool)

    parser.add_argument("--test", default=False, type=bool)    
    parser.add_argument("--test_zoo", default=False, type=bool)    
    parser.add_argument("--weight_path", default='/home/zli911/imitation/expert_files/', type=str)
    parser.add_argument("--model_name", default='best_model.zip', type=str)


    parser.add_argument("--save_iq", default=False, type=bool)   
    
    args = parser.parse_args()
    print(args)
    if args.save_iq:
        save_expert_trajectory_IQ(args)
    elif args.test_zoo:
        test_zoo_expert(args)
    elif args.test:
        assert args.weight_path is not None, "Please provide the path to the pretrained expert!"
        test_expert(args)
    else:
        train_expert(args)

