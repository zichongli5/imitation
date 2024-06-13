import numpy as np
import gymnasium as gym
import os
import argparse
import torch
import torch.nn as nn
import cv2

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from rl_zoo3 import ALGOS
from rl_zoo3.utils import get_wrapper_class
from imitation.data import rollout, types
from torch.utils.data import DataLoader, RandomSampler
from imitation.algorithms import bc

from utils import pearson_correlation ,RewardDataset, create_env, evaluate, get_saved_hyperparams, preprocess_hyperparams, load_pretrained_expert, load_stats_to_env
from networks import SACBCPolicy, RewardNet_from_policy, BC_base, PPOBCPolicy, PPOBCCNNPolicy, BC_AT

BC_POLICY_DICT = {'sac':SACBCPolicy, 'ppo_s':PPOBCPolicy,'ppo':PPOBCCNNPolicy, 'base': BC_base}

def train_bc(env, expert, expert_episodes=1, seed=0, save=True, args=None):
    '''
    Train behavior cloning agent
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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
    # print(len(transitions))

    policy = BC_POLICY_DICT[args.bc_algo](env.observation_space, env.action_space, lr_schedule=lambda _: torch.finfo(torch.float32).max)
    print(policy)
    policy.to(device)
    if args.adaptive_temp:
        bc_cls = BC_AT
        bc_trainer = bc_cls(observation_space=env.observation_space, 
                       action_space=env.action_space,
                       demonstrations=transitions,
                       policy=policy,
                       batch_size=args.bc_batch_size,
                       rng=rng,
                       device=device,
                       ent_weight=args.bc_ent_weight,
                       l2_weight=args.bc_l2_weight,
                       N=len(transitions),
                       rho=args.rho,
                       tau_init=1.0,
                       tau_min=0.1,
                       tau_max=5.0
                       )
        
    else:
        bc_cls = bc.BC
        bc_trainer = bc_cls(observation_space=env.observation_space, 
                        action_space=env.action_space,
                        demonstrations=transitions,
                        policy=policy,
                        batch_size=args.bc_batch_size,
                        rng=rng,
                        device=device,
                        ent_weight=args.bc_ent_weight,
                        l2_weight=args.bc_l2_weight,
                        )
       
    bc_trainer.train(n_epochs=args.bc_epochs)
    
    print('Evaluate the Behavior Cloning Agent!!!')
    # total_returns, total_timesteps = evaluate_policy(bc_trainer.policy, env, 10)
    # print("Reward:", reward)
    total_returns, total_timesteps = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=args.eval_steps, return_episode_rewards=True)
    print('BC Return: {} +/- {}'.format(np.mean(total_returns), np.std(total_returns)))
    print('BC Timesteps: {} +/- {}'.format(np.mean(total_timesteps), np.std(total_timesteps)))


    # Save the trained policy
    if args.save_path is not None and save:
        save_path = os.path.join(args.save_path, args.env_id, 'bc_{}'.format(args.bc_algo))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        bc_trainer.policy.save(os.path.join(save_path, 'bc_policy_{}_{}_{}_{}_{}_{}_seed_{}{}.pkl'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.seed, '_AT' if args.adaptive_temp else '')))
        print('Behavior Cloning Policy saved!')
    if args.save_result is not None and save:
        result_path = os.path.join(args.save_result, args.env_id, 'bc_{}'.format(args.bc_algo))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        np.savez(os.path.join(result_path, 'bc_result_{}_{}_{}_{}_{}_{}_{}_seed_{}{}'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.rho, args.seed, '_AT' if args.adaptive_temp else ''))
                 , returns=total_returns, timesteps=total_timesteps)
        print('Behavior Cloning Result saved!')
        # save the logit and tau
        result_path = os.path.join(args.save_result, args.env_id, 'bc_tau_logit')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        transitions_rew = rollout.flatten_trajectories_with_rew(rollouts)
        reward_net_bc = RewardNet_from_policy(bc_trainer.policy.cpu(), alpha=1.0)
        reward_dataloader = torch.utils.data.DataLoader(RewardDataset(transitions_rew), batch_size=256, shuffle=False)
        bc_logit = []
        for _ in range(1):
            for obs, next_obs, action, done, info, rew in reward_dataloader:
                predicted_reward_bc_logit = reward_net_bc.get_logits(obs, action, next_obs, done)
                bc_logit.append(predicted_reward_bc_logit.detach().cpu())
        bc_logit = torch.cat(bc_logit, dim=0).cpu()
        if args.adaptive_temp:
            np.savez(os.path.join(result_path, 'logittau_{}_{}_{}_{}_{}_{}_{}_seed_{}{}'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.rho, args.seed, '_AT' if args.adaptive_temp else ''))
            , logit = bc_logit, tau = bc_trainer.tau.detach().cpu(), tau_list = torch.cat(bc_trainer.tau_list,0))
        else:
            np.savez(os.path.join(result_path, 'logittau_{}_{}_{}_{}_{}_{}_seed_{}{}'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.seed, '_AT' if args.adaptive_temp else ''))
                , logit = bc_logit)

        
    # if not args.adaptive_temp:
    #     rng=np.random.default_rng(seed)
    #     policy_at = BC_POLICY_DICT[args.bc_algo](env.observation_space, env.action_space, lr_schedule=lambda _: torch.finfo(torch.float32).max)
    #     policy_at.to(device)
    #     bc_trainer_at = BC_AT(observation_space=env.observation_space, 
    #                     action_space=env.action_space,
    #                     demonstrations=transitions,
    #                     policy=policy_at,
    #                     batch_size=args.bc_batch_size,
    #                     rng=rng,
    #                     device=device,
    #                     ent_weight=args.bc_ent_weight,
    #                     l2_weight=args.bc_l2_weight,
    #                     N=len(transitions),
    #                     rho=args.rho,
    #                     tau_init=1.0,
    #                     tau_min=0.1,
    #                     tau_max=5.0
    #                     )
    #     bc_trainer_at.train(n_epochs=args.bc_epochs)
    #     tau = bc_trainer_at.tau.detach().cpu()
    #     # print(tau[:10])
    #     print('Evaluate the Behavior Cloning Agent AAAAAATTTTTT!!!')
    #     # total_returns, total_timesteps = evaluate_policy(bc_trainer.policy, env, 10)
    #     # print("Reward:", reward)
    #     total_returns, total_timesteps = evaluate_policy(bc_trainer_at.policy, env, n_eval_episodes=args.eval_steps, return_episode_rewards=True)
    #     print('BC Return: {} +/- {}'.format(np.mean(total_returns), np.std(total_returns)))
    #     print('BC Timesteps: {} +/- {}'.format(np.mean(total_timesteps), np.std(total_timesteps)))
    # else:
    #     tau = bc_trainer.tau.detach().cpu()
    #     bc_trainer_at = bc_trainer
        
    # transitions_rew = rollout.flatten_trajectories_with_rew(rollouts)
    # reward_net_bc = RewardNet_from_policy(bc_trainer.policy.cpu(), alpha=1.0)
    # reward_net_bc_at = RewardNet_from_policy(bc_trainer_at.policy.cpu(), alpha=1.0)
    # reward_net_expert = RewardNet_from_policy(expert.policy.cpu(), alpha=1.0)


    # reward_dataloader = torch.utils.data.DataLoader(RewardDataset(transitions_rew), batch_size=256, shuffle=False)
    # true_rew = []
    # bc_rew = []
    # bc_rew_at = []
    # bc_rew_logit = []
    # bc_rew_at_logit = []
    # expert_rew = []
    # expert_rew_p = []
    # rw_all = 0
    # # args.rw_epochs = 1
    # for epoch in range(1):
    #     epoch_loss = 0
    #     for obs, next_obs, action, done, info, rew in reward_dataloader:
    #         predicted_reward_bc = reward_net_bc(obs, action, next_obs, done)
    #         predicted_reward_bc_at = reward_net_bc_at(obs, action, next_obs, done)
    #         predicted_reward_bc_logit = reward_net_bc.get_logits(obs, action, next_obs, done)
    #         predicted_reward_bc_at_logit = reward_net_bc_at.get_logits(obs, action, next_obs, done)

    #         predicted_reward_expert = reward_net_expert(obs, action, next_obs, done)
            
    #         with torch.no_grad():
    #             v_expert = expert.policy.predict_values(obs).squeeze()
    #             next_v_expert = expert.policy.predict_values(next_obs).squeeze()
    #             y = (1 - done.float().squeeze()) * expert.gamma * next_v_expert
    #             reward_target_expert = (v_expert.squeeze() - y.squeeze()) * 100
    #             # print(reward_target_expert.size(),v_expert.size(), next_v_expert.size(), y.size(), done.size())
    #             # print(reward_target)

    #         true_rew.append(rew.detach())
    #         bc_rew.append(predicted_reward_bc.detach())
    #         bc_rew_at.append(predicted_reward_bc_at.detach())
    #         bc_rew_logit.append(predicted_reward_bc_logit.detach())
    #         bc_rew_at_logit.append(predicted_reward_bc_at_logit.detach())
    #         expert_rew.append(reward_target_expert.detach())
    #         expert_rew_p.append(predicted_reward_expert.detach())

    # # calculate correlation between predicted reward and true reward
    # true_rew = torch.cat(true_rew, dim=0).cpu()
    # bc_rew = torch.cat(bc_rew, dim=0).cpu()
    # bc_rew_at = torch.cat(bc_rew_at, dim=0).cpu()
    # bc_rew_logit = torch.cat(bc_rew_logit, dim=0).cpu()
    # bc_rew_at_logit = torch.cat(bc_rew_at_logit, dim=0).cpu()
    # expert_rew = torch.cat(expert_rew, dim=0).cpu()
    # expert_rew_p = torch.cat(expert_rew_p, dim=0).cpu()
    # print('Correlation between true reward and BC reward: {}'.format(pearson_correlation(true_rew, bc_rew)))
    # print('Correlation between true reward and BC reward AT: {}'.format(pearson_correlation(true_rew, bc_rew_at)))
    # print('Correlation between true reward and BC reward logit: {}'.format(pearson_correlation(true_rew, bc_rew_logit)))
    # print('Correlation between true reward and BC reward AT logit: {}'.format(pearson_correlation(true_rew, bc_rew_at_logit)))
    # print('Correlation between true reward and expert reward: {}'.format(pearson_correlation(true_rew, expert_rew)))
    # print('Correlation between true reward and expert predicted reward: {}'.format(pearson_correlation(true_rew, expert_rew_p)))

    # # save the logit and tau
    # result_path = os.path.join(args.save_result, args.env_id, 'bc_corr')
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # np.savez(os.path.join(result_path, 'corr_{}_{}_{}_{}_{}_{}_seed_{}'.format(expert_episodes, args.bc_epochs, args.bc_batch_size, args.bc_hidden_size, args.bc_ent_weight, args.bc_l2_weight, args.seed))
    #              , tau = tau, logit_at = bc_rew_at_logit, logit = bc_rew_logit, true_rew = true_rew, expert_rew = expert_rew, expert_rew_p = expert_rew_p, bc_rew = bc_rew, bc_rew_at = bc_rew_at)


    return bc_trainer, total_returns, total_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="PongNoFrameskip-v4")
    parser.add_argument("--algo", default='ppo', type=str)
    parser.add_argument("--config_path", default='/home/zli911/imitation/expert_files/rl-trained-agents/', type=str)
    parser.add_argument("--seed", default=0, type=int)
   
    parser.add_argument("--test_zoo", default=True, type=bool)  
    parser.add_argument("--eval_steps", default=5, type=int)
    parser.add_argument("--weight_path", default='/home/zli911/imitation/expert_files/', type=str)
    parser.add_argument("--model_name", default='best_model.zip', type=str)

    parser.add_argument("--save_path", default='/home/zli911/imitation/weight_files/', type=str)
    parser.add_argument("--save_result", default='/home/zli911/imitation/result_files/', type=str)

    parser.add_argument("--adaptive_temp", default=0, type=int)
    parser.add_argument("--rho", default=0.5, type=float)


    parser.add_argument("--expert_episodes", default=1, type=int)
    parser.add_argument("--bc_algo", default='ppo', type=str)
    parser.add_argument("--bc_epochs", default=100, type=int)
    parser.add_argument("--bc_batch_size", default=64, type=int)
    parser.add_argument("--bc_hidden_size", default=64, type=int)
    parser.add_argument("--bc_ent_weight", default=0.00, type=float)
    parser.add_argument("--bc_l2_weight", default=0.00, type=float)
    
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
        print(config_path)
        hyperparams = get_saved_hyperparams(config_path)

        # Create environment
        print(hyperparams)
        env_wrapper = get_wrapper_class(hyperparams)
        eval_env = create_env(env_id, n_envs=1, norm_obs=hyperparams['norm_obs'], norm_reward=False, seed=seed, stats_path=None, env_wrapper=env_wrapper, manual_load=True, frame_num=hyperparams['frame_stack'])
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