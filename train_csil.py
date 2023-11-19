import numpy as np
import gymnasium as gym
import os
import argparse
import torch

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from rl_zoo3 import ALGOS
from rl_zoo3.utils import get_wrapper_class

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

from utils import create_env, evaluate, get_saved_hyperparams, wrap_bc_policy, preprocess_hyperparams, load_pretrained_expert, load_stats_to_env
from train_bc import train_bc
from networks import RewardNet_from_policy

def train_csil(args):
    
    assert os.path.exists(args.weight_path), "Wrong path to the pretrained expert!"
    env_id = args.env_id
    algo = args.algo
    config_path = args.config_path
    seed = args.seed


    config_expert_path = os.path.join(config_path, 'ppo', env_id+'_1', env_id)
    if args.test_zoo:
        # Get the right path
        zoo_path = '/home/zli911/imitation/expert_files/rl-trained-agents'
        stats_path = os.path.join(zoo_path, 'ppo', env_id+'_1', env_id, 'vecnormalize.pkl')
        model_path = os.path.join(zoo_path, 'ppo', env_id+'_1',env_id+'_gymnasium.zip')
        # if model path doesn't exist, load the old pretrained expert and re-save it
        if not os.path.exists(model_path) and args.test_zoo:
            print('No model path provided, load the old pretrained expert and resave it!')
            model_path_old = os.path.join(zoo_path, algo, env_id+'_1',env_id+'.zip')
            expert = ALGOS[algo].load(model_path_old, custom_objects=custom_objects)
            expert.save(model_path)
            print('Expert model re-saved!')
    else:
        # Get the right path
        model_path = os.path.join(args.weight_path, env_id, 'ppo', args.model_name)
        stats_path = os.path.join(args.weight_path, env_id, 'ppo', env_id+'_stats.pkl')
    
    # Load hyperparameters from yaml file
    hyperparams_expert = get_saved_hyperparams(config_expert_path)

    # Create environment
    print(hyperparams_expert)
    env_wrapper = get_wrapper_class(hyperparams_expert)
    eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams_expert['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path, env_wrapper=env_wrapper, manual_load=args.test_zoo)
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    } if args.test_zoo else None
    
    # Load model
    expert = ALGOS['ppo'].load(model_path, custom_objects=custom_objects)
    print('Expert model loaded!')


    # Train behavior cloning agent
    bc_trainer, bc_returns, bc_timesteps = train_bc(eval_env, expert, expert_episodes=args.expert_episodes, seed=0, save=False, args=args)

    # Wrap the bc policy into a reward network
    reward_net = RewardNet_from_policy(bc_trainer.policy, alpha=args.bc_ent_weight)
    print('Reward network created!')

    # Wrap the environment with the reward network
    n_envs = 1
    env_train = create_env(env_id, n_envs=n_envs, norm_obs=hyperparams_expert['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path, env_wrapper=env_wrapper, manual_load=args.test_zoo,
                           reward_wrap=reward_net)
    print('Environment wrapped with reward network!')

    # Create CSIL agent
    # config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
    # hyperparams = get_saved_hyperparams(config_path)
    csil_trainer = ALGOS[algo](policy='MlpPolicy', env=env_train, verbose=1, seed=seed, 
                                batch_size=args.csil_batch_size, buffer_size=args.csil_buffer_size,
                                tau=args.csil_tau, learning_rate=args.csil_lr, train_freq=args.csil_train_freq,
                                gradient_steps=args.csil_gradient_steps)
    # csil_trainer = ALGOS['ppo'](policy='MlpPolicy', env=eval_env, verbose=1, seed=seed)
    
    # csil_trainer.policy.actor = wrap_bc_policy(bc_trainer.policy)
    # csil_trainer.actor = csil_trainer.policy.actor
    if args.bc_algo == 'sac':
        csil_trainer.policy = bc_trainer.policy
        print('Initialize CSIL agent with BC policy')

    # print(csil_trainer.policy.actor)

    # Generate callbacks
    save_path = os.path.join(args.save_path, args.env_id, 'csil')
    # eval_env_rw = RewardVecEnvWrapper(eval_env, reward_net.predict_processed)
    eval_callback = EvalCallback(env_train, best_model_save_path=save_path, log_path=save_path, eval_freq=max(args.eval_freq // n_envs,1), n_eval_episodes = 10, deterministic=True, render=False)
    # eval_callback = None

    # Train CSIL agent
    csil_trainer.learn(total_timesteps=args.csil_timesteps, callback=eval_callback)

    # Evaluate the trained agent
    print('Evaluate the trained CSIL agent')
    csil_returns, csil_timesteps = evaluate(eval_env, csil_trainer.policy, num_episodes=args.eval_steps)
    print('CSIL Return: {} +/- {}'.format(np.mean(csil_returns), np.std(csil_returns)))
    print('CSIL Timesteps: {} +/- {}'.format(np.mean(csil_timesteps), np.std(csil_timesteps)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument("--env_id", help="Name of environment", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--algo", default='sac', type=str, help="CSIL agent to use")
    parser.add_argument("--config_path", default='/home/zli911/imitation/expert_files/rl-trained-agents/', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=100000, type=int)

    # Expert parameters
    parser.add_argument("--test_zoo", default=True, type=bool)  
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--weight_path", default='/home/zli911/imitation/expert_files/', type=str)
    parser.add_argument("--model_name", default='best_model.zip', type=str)
    parser.add_argument("--expert_episodes", default=1, type=int)

    parser.add_argument("--save_path", default='/home/zli911/imitation/weight_files/', type=str)
    parser.add_argument("--save_result", default='/home/zli911/imitation/result_files/', type=str)

    # BC parameters
    parser.add_argument("--bc_epochs", default=100, type=int)
    parser.add_argument("--bc_algo", default='sac', type=str)
    parser.add_argument("--bc_batch_size", default=32, type=int)
    parser.add_argument("--bc_hidden_size", default=256, type=int)
    parser.add_argument("--bc_ent_weight", default=0.05, type=float)
    parser.add_argument("--bc_l2_weight", default=0.01, type=float)

    # CSIL parameters
    parser.add_argument("--csil_timesteps", default=1000000, type=int)
    parser.add_argument("--csil_batch_size", default=256, type=int)
    parser.add_argument("--csil_buffer_size", default=300000, type=int)
    parser.add_argument("--csil_tau", default=0.02, type=int)
    parser.add_argument("--csil_lr", default=3e-4, type=int)
    parser.add_argument("--csil_train_freq", default=64, type=int)
    parser.add_argument("--csil_gradient_steps", default=64, type=int)


    
    args = parser.parse_args()
    print(args)

    train_csil(args)