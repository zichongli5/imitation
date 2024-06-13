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

from imitation.algorithms import bc, base
from imitation.data import rollout
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.reward_nets import BasicRewardNet
import sys

from utils import RewardDataset, pearson_correlation, create_env, evaluate, get_saved_hyperparams, wrap_bc_policy, preprocess_hyperparams, load_pretrained_expert, load_stats_to_env, wrap_iq_agent
from train_bc import train_bc
from networks import RewardNet_from_policy, SACBCPolicy
from omegaconf import DictConfig, OmegaConf
import copy

def train_cqil(args):
    
    assert os.path.exists(args.weight_path), "Wrong path to the pretrained expert!"
    env_id = args.env_id
    algo = args.algo
    config_path = args.config_path
    seed = args.seed
    args.bc_algo = algo
    print(args)

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
    bc_trainer, bc_returns, bc_timesteps = train_bc(eval_env, expert, expert_episodes=args.expert_episodes, seed=seed, save=False, args=args)
    reward_net_bc = RewardNet_from_policy(bc_trainer.policy, alpha=1.0)
    reward_net_expert = RewardNet_from_policy(expert.policy, alpha=1.0)

    # Load IQ-learn model
    sys.path.insert(0, '/home/zli911/imitation/baselines/IQ-Learn/iq_learn')
    from agent import make_agent
    import utils_iq

    iq_path = args.iq_path
    iq_model_path = os.path.join(iq_path, 'results')
    yaml_path = os.path.join(iq_path, 'conf', 'config_all.yaml')

    config_iq = OmegaConf.load(yaml_path)
    config_iq.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    agent = make_agent(eval_env, config_iq)
    agent.load(iq_model_path, suffix = f'_iq_{env_id}_{args.expert_episodes}')
    iq_agent = wrap_iq_agent(agent)

    # Evaluate IQ-learn agent
    print('Evaluate the trained IQ-learn agent')
    iq_returns, iq_timesteps = evaluate(eval_env, iq_agent, num_episodes=args.eval_steps)
    # iq_returns, iq_timesteps = utils_iq.utils.evaluate(agent, eval_env, num_episodes=args.eval_steps)
    print('IQ-learn Return: {} +/- {}'.format(np.mean(iq_returns), np.std(iq_returns)))
    print('IQ-learn Timesteps: {} +/- {}'.format(np.mean(iq_timesteps), np.std(iq_timesteps)))         
    # Q = agent.critic
    # IQ_actor = agent.actor
    
    # random_policy = SACBCPolicy(eval_env.observation_space, eval_env.action_space, lr_schedule=lambda _: torch.finfo(torch.float32).max, net_arch=[args.bc_hidden_size, args.bc_hidden_size])
    # Rollout IQ agent for training rewawrd network
    rng=np.random.default_rng(seed)
    rollouts = rollout.rollout(iq_agent.predict, eval_env, 
                               rollout.make_sample_until(min_timesteps=None, min_episodes=args.iq_episodes),
                               unwrap=False,
                               verbose=True,
                               rng=rng,
                               )
    stats = rollout.rollout_stats(rollouts)
    print(f"Rollout stats: {stats}")
    transitions = rollout.flatten_trajectories_with_rew(rollouts)
    # print(transitions)

    # Create reward network
    reward_net = BasicRewardNet(eval_env.observation_space, eval_env.action_space, use_state=True, use_action=True, use_next_state=True, hid_sizes=(args.rw_hidden_size,args.rw_hidden_size))

    # Train reward network
    print('-----------------Start training reward network-----------------')
    reward_dataloader = torch.utils.data.DataLoader(RewardDataset(transitions), batch_size=args.rw_batch_size, shuffle=True)
    reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=args.rw_lr)
    criterion = torch.nn.MSELoss()
    true_rew = []
    bc_rew = []
    iq_rew = []
    expert_rew = []
    expert_rew_p = []
    rw_all = 0
    # args.rw_epochs = 1
    for epoch in range(args.rw_epochs):
        epoch_loss = 0
        for obs, next_obs, action, done, info, rew in reward_dataloader:
            # print(obs, next_obs, action, done, info)
            reward_optimizer.zero_grad()
            predicted_reward = reward_net(obs, action, next_obs, done)
            predicted_reward_bc = reward_net_bc(obs, action, next_obs, done)
            predicted_reward_expert = reward_net_expert(obs, action, next_obs, done)
            
            with torch.no_grad():
                q = agent.infer_q(obs, action)
                q = torch.from_numpy(q).squeeze(-1)
                next_v = agent.infer_v(next_obs).copy()
                y = (1 - done.float()) * agent.gamma * next_v
                reward_target = (q - y)
                # print(reward_target)
            with torch.no_grad():
                v_expert = expert.policy.predict_values(obs).squeeze()
                next_v_expert = expert.policy.predict_values(next_obs).squeeze()
                y = (1 - done.float().squeeze()) * expert.gamma * next_v_expert
                reward_target_expert = (v_expert.squeeze() - y.squeeze()) * 100
                # print(reward_target_expert.size(),v_expert.size(), next_v_expert.size(), y.size(), done.size())
                # print(reward_target)
            if epoch == 0:
                true_rew.append(rew.detach())
                bc_rew.append(predicted_reward_bc.detach())
                iq_rew.append(reward_target.detach())
                expert_rew.append(reward_target_expert.detach())
                expert_rew_p.append(predicted_reward_expert.detach())
                rw_all += reward_target.sum()

            loss = criterion(predicted_reward, reward_target.detach())
            loss.backward()
            reward_optimizer.step()
            epoch_loss += loss.item()

        print("RM epoch", epoch, epoch_loss / len(reward_dataloader))
    print('-----------------Reward network training finished-----------------')
    # calculate correlation between predicted reward and true reward
    true_rew = torch.cat(true_rew, dim=0)
    bc_rew = torch.cat(bc_rew, dim=0)
    iq_rew = torch.cat(iq_rew, dim=0)
    expert_rew = torch.cat(expert_rew, dim=0)
    expert_rew_p = torch.cat(expert_rew_p, dim=0)
    print('Correlation between true reward and BC reward: {}'.format(pearson_correlation(true_rew, bc_rew)))
    print('Correlation between true reward and IQ reward: {}'.format(pearson_correlation(true_rew, iq_rew)))
    print('Correlation between true reward and expert reward: {}'.format(pearson_correlation(true_rew, expert_rew)))
    print('Correlation between true reward and expert predicted reward: {}'.format(pearson_correlation(true_rew, expert_rew_p)))
    # print(rw_all)
    # print(true_rew[:100])
    # print(bc_rew[:100])
    # print(iq_rew[:100])
    # print(expert_rew[:100])
    # print(expert_rew_p[:100])

    # freeze the reward network
    for param in reward_net.parameters():
        param.requires_grad = False

    # Wrap the environment with the reward network
    n_envs = 1
    env_train = create_env(env_id, n_envs=n_envs, norm_obs=hyperparams_expert['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path, env_wrapper=env_wrapper, manual_load=args.test_zoo
                           , reward_wrap=reward_net)
    # env_train = RewardVecEnvWrapper(env_train, reward_net.predict_processed, 0)
    print('Environment wrapped with reward network!')

    # Create cqil agent
    # config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
    # hyperparams = get_saved_hyperparams(config_path)
    if algo == 'sac':
        cqil_trainer = ALGOS[algo](policy='MlpPolicy', env=env_train, verbose=1, seed=seed, 
                                    batch_size=args.cqil_batch_size, buffer_size=args.cqil_buffer_size,
                                    tau=args.cqil_tau, learning_rate=args.cqil_lr, train_freq=args.cqil_train_freq,
                                    gradient_steps=args.cqil_gradient_steps, policy_kwargs=dict(net_arch=[args.bc_hidden_size, args.bc_hidden_size]))
    elif algo == 'ppo':
        cqil_trainer = ALGOS[algo](policy='MlpPolicy', env=env_train, learning_rate=args.cqil_lr, 
                                        n_steps=args.cqil_n_steps, batch_size=args.cqil_batch_size, 
                                        n_epochs=args.cqil_n_epochs, clip_range=args.cqil_clip_range,
                                        policy_kwargs=dict(net_arch=[args.bc_hidden_size, args.bc_hidden_size]), 
                                        verbose=1)


    cqil_trainer.policy.load_state_dict(bc_trainer.policy.state_dict())
    print('Initialize with behavior cloning policy!')
    cqil_returns, cqil_timesteps = evaluate(eval_env, cqil_trainer.policy, num_episodes=args.eval_steps)
    print('Initial CQIL Return: {} +/- {}'.format(np.mean(cqil_returns), np.std(cqil_returns)))
    print('Initial CQIL Timesteps: {} +/- {}'.format(np.mean(cqil_timesteps), np.std(cqil_timesteps)))

    # Generate callbacks
    save_path = os.path.join(args.save_path, args.env_id, 'cqil')
    eval_callback = EvalCallback(env_train, best_model_save_path=save_path, log_path=save_path, eval_freq=max(args.eval_freq // n_envs,1), n_eval_episodes = 10, deterministic=True, render=False)
    # eval_callback = None

    # Train cqil agent
    cqil_trainer.learn(total_timesteps=args.cqil_timesteps, callback=eval_callback)

    # Evaluate the trained agent
    print('Evaluate the trained cqil agent')
    cqil_returns, cqil_timesteps = evaluate(eval_env, cqil_trainer.policy, num_episodes=100)
    print('cqil Return: {} +/- {}'.format(np.mean(cqil_returns), np.std(cqil_returns)))
    print('cqil Timesteps: {} +/- {}'.format(np.mean(cqil_timesteps), np.std(cqil_timesteps)))

    # Save the trained policy
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, args.env_id, 'cqil_3')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cqil_trainer.policy.save(os.path.join(save_path, 'cqil_policy_{}_{}_{}_{}_{}_{}_seed_{}.pkl'.format(args.expert_episodes, args.rw_epochs, args.rw_batch_size, args.rw_hidden_size, args.rw_lr, args.iq_episodes, args.seed)))
        print('CQIL Policy saved!')
    if args.save_result is not None:
        result_path = os.path.join(args.save_result, args.env_id, 'cqil_3')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        np.savez(os.path.join(result_path, 'cqil_result_{}_{}_{}_{}_{}_{}_seed_{}'.format(args.expert_episodes, args.rw_epochs, args.rw_batch_size, args.rw_hidden_size, args.rw_lr, args.iq_episodes, args.seed))
                 , returns=cqil_returns, timesteps=cqil_timesteps, bc_returns=bc_returns, bc_timesteps=bc_timesteps, iq_returns=iq_returns, iq_timesteps=iq_timesteps)
        print('CQIL Result saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument("--env_id", help="Name of environment", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--algo", default='ppo', type=str, help="CQIL agent to use")
    parser.add_argument("--config_path", default='/home/zli911/imitation/expert_files/rl-trained-agents/', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=100000, type=int)

    # Expert parameters
    parser.add_argument("--test_zoo", default=True, type=bool)  
    parser.add_argument("--eval_steps", default=50, type=int)
    parser.add_argument("--weight_path", default='/home/zli911/imitation/expert_files/', type=str)
    parser.add_argument("--model_name", default='best_model.zip', type=str)
    parser.add_argument("--expert_episodes", default=1, type=int)

    parser.add_argument("--save_path", default='/home/zli911/imitation/weight_files/', type=str)
    parser.add_argument("--save_result", default='/home/zli911/imitation/result_files/', type=str)

    # BC parameters
    parser.add_argument("--bc_epochs", default=50, type=int)
    parser.add_argument("--bc_algo", default='sac', type=str)
    parser.add_argument("--bc_batch_size", default=64, type=int)
    parser.add_argument("--bc_hidden_size", default=64, type=int)
    parser.add_argument("--bc_ent_weight", default=0.01, type=float)
    parser.add_argument("--bc_l2_weight", default=0.01, type=float)

    # Reward Net training parameters
    parser.add_argument("--iq_episodes", default=10, type=int)
    parser.add_argument("--rw_epochs", default=20, type=int)
    parser.add_argument("--rw_batch_size", default=64, type=int)
    parser.add_argument("--rw_lr", default=1e-5, type=float)
    parser.add_argument("--rw_hidden_size", default=64, type=int)

    # IQ learn parameters
    parser.add_argument("--iq_path", default='/home/zli911/imitation/baselines/IQ-Learn/iq_learn/', type=str)

    # cqil parameters 
    parser.add_argument("--cqil_timesteps", default=1000000, type=int)
    parser.add_argument("--cqil_batch_size", default=256, type=int)
    parser.add_argument("--cqil_lr", default=3e-5, type=int)

    # for SAC
    parser.add_argument("--cqil_buffer_size", default=300000, type=int)
    parser.add_argument("--cqil_tau", default=0.02, type=int)
    parser.add_argument("--cqil_train_freq", default=64, type=int)
    parser.add_argument("--cqil_gradient_steps", default=64, type=int)

    # for PPO
    parser.add_argument("--cqil_n_steps", default=2048, type=int)
    parser.add_argument("--cqil_n_epochs", default=10, type=int)
    parser.add_argument("--cqil_clip_range", default=0.2, type=float)


    
    args = parser.parse_args()

    train_cqil(args)