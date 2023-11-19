import numpy as np
import gymnasium as gym
import os
import argparse
import torch

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from rl_zoo3 import ALGOS
from rl_zoo3.utils import get_wrapper_class

from imitation.algorithms import bc
from imitation.data import rollout

from utils import create_env, evaluate, get_saved_hyperparams, preprocess_hyperparams, load_pretrained_expert, load_stats_to_env
from networks import SACBCPolicy, RewardNet_from_policy

BC_POLICY_DICT = {'sac':SACBCPolicy, 'ppo':ActorCriticPolicy}

def train_bc(env, expert, expert_episodes=1, seed=0, save=True, args=None):
    '''
    Train behavior cloning agent
    '''
    # Collect expert trajectories
    rng=np.random.default_rng(seed)
    rollouts = rollout.rollout(expert, env, 
                               rollout.make_sample_until(min_timesteps=None, min_episodes=expert_episodes),
                               unwrap=False,
                               verbose=True,
                               rng=rng,
                               deterministic_policy=True
                               )
    # Truncate rollouts based on expert_episodes
    rollouts = rollouts[:expert_episodes]
    stats = rollout.rollout_stats(rollouts)
    print(f"Rollout stats: {stats}")

    transitions = rollout.flatten_trajectories(rollouts)

    policy = BC_POLICY_DICT[args.bc_algo](env.observation_space, env.action_space, lr_schedule=lambda _: torch.finfo(torch.float32).max, net_arch=[args.bc_hidden_size, args.bc_hidden_size])
    bc_trainer = bc.BC(observation_space=env.observation_space, 
                       action_space=env.action_space,
                       demonstrations=transitions,
                       policy=policy,
                       batch_size=args.bc_batch_size,
                       rng=rng,
                       ent_weight=args.bc_ent_weight,
                       l2_weight=args.bc_l2_weight,
                       )
    bc_trainer.train(n_epochs=args.bc_epochs)
    
    print('Evaluate the Behavior Cloning Agent!!!')
    # total_returns, total_timesteps = evaluate_policy(bc_trainer.policy, env, 10)
    # print("Reward:", reward)
    total_returns, total_timesteps = evaluate(env, bc_trainer.policy, num_episodes=args.eval_steps, deterministic=True)
    print('BC Return: {} +/- {}'.format(np.mean(total_returns), np.std(total_returns)))
    print('BC Timesteps: {} +/- {}'.format(np.mean(total_timesteps), np.std(total_timesteps)))

    # Save the trained policy
    if args.save_path is not None and save:
        save_path = os.path.join(args.save_path, args.env_id, 'bc_{}'.format(args.bc_algo))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        bc_trainer.policy.save(os.path.join(save_path, 'bc_policy_{}_{}_{}_{}_{}_{}_seed_{}.pkl'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.seed)))
        print('Behavior Cloning Policy saved!')
    if args.save_result is not None and save:
        result_path = os.path.join(args.save_result, args.env_id, 'bc_{}'.format(args.bc_algo))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        np.savez(os.path.join(result_path, 'bc_result_{}_{}_{}_{}_{}_{}_seed_{}'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.seed))
                 , returns=total_returns, timesteps=total_timesteps)
        print('Behavior Cloning Result saved!')
    return bc_trainer, total_returns, total_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--algo", default='ppo', type=str)
    parser.add_argument("--config_path", default='/home/zli911/imitation/expert_files/rl-trained-agents/', type=str)
    parser.add_argument("--seed", default=0, type=int)
   
    parser.add_argument("--test_zoo", default=True, type=bool)  
    parser.add_argument("--eval_steps", default=100, type=int)
    parser.add_argument("--weight_path", default='/home/zli911/imitation/expert_files/', type=str)
    parser.add_argument("--model_name", default='best_model.zip', type=str)

    parser.add_argument("--save_path", default='/home/zli911/imitation/weight_files/', type=str)
    parser.add_argument("--save_result", default='/home/zli911/imitation/result_files/', type=str)


    parser.add_argument("--expert_episodes", default=1, type=int)
    parser.add_argument("--bc_algo", default='sac', type=str)
    parser.add_argument("--bc_epochs", default=100, type=int)
    parser.add_argument("--bc_batch_size", default=32, type=int)
    parser.add_argument("--bc_hidden_size", default=256, type=int)
    parser.add_argument("--bc_ent_weight", default=0.00, type=float)
    parser.add_argument("--bc_l2_weight", default=0.01, type=float)
    
    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.weight_path), "Wrong path to the pretrained expert!"
    env_id = args.env_id
    algo = args.algo
    config_path = args.config_path
    seed = args.seed

    if args.test_zoo:
        # Get the right path
        zoo_path = '/home/zli911/imitation/expert_files/rl-trained-agents'
        config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
        stats_path = os.path.join(zoo_path, algo, env_id+'_1', env_id, 'vecnormalize.pkl')
        model_path = os.path.join(zoo_path, algo, env_id+'_1',env_id+'_gymnasium.zip')

        # Load hyperparameters from yaml file
        hyperparams = get_saved_hyperparams(config_path)

        # Create environment
        print(hyperparams)
        env_wrapper = get_wrapper_class(hyperparams)
        eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams['norm_obs'], norm_reward=False, seed=seed, stats_path=None, env_wrapper=env_wrapper)
        eval_env = load_stats_to_env(stats_path, eval_env)
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    else:
        # Get the right path
        config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
        model_path = os.path.join(args.weight_path, env_id, algo, args.model_name)
        stats_path = os.path.join(args.weight_path, env_id, algo, env_id+'_stats.pkl')

        # Load hyperparameters from yaml file
        hyperparams = get_saved_hyperparams(config_path)

        # Create environment
        env_wrapper = get_wrapper_class(hyperparams)
        eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path, env_wrapper=env_wrapper)
        custom_objects = None
    
    # Load model
    # if model path doesn't exist, load the old pretrained expert and re-save it
    if not os.path.exists(model_path) and args.test_zoo:
        print('No model path provided, load the old pretrained expert and resave it!')
        model_path_old = os.path.join(zoo_path, algo, env_id+'_1',env_id+'.zip')
        expert = ALGOS[algo].load(model_path_old, custom_objects=custom_objects)
        expert.save(model_path)
        print('Expert model re-saved!')

    expert = ALGOS[algo].load(model_path, custom_objects=custom_objects)
    print('Expert model loaded!')

    # Train behavior cloning agent
    bc_trainer = train_bc(eval_env, expert, expert_episodes=args.expert_episodes, seed=0, args=args)