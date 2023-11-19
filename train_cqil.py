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

from utils import RewardDataset, create_env, evaluate, get_saved_hyperparams, wrap_bc_policy, preprocess_hyperparams, load_pretrained_expert, load_stats_to_env, wrap_iq_agent
from train_bc import train_bc
from networks import RewardNet_from_policy
from omegaconf import DictConfig, OmegaConf
import copy

def train_cqil(args):
    
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
    
    rng=np.random.default_rng(seed)
    rollouts = rollout.rollout(iq_agent.predict, eval_env, 
                               rollout.make_sample_until(min_timesteps=None, min_episodes=args.iq_episodes),
                               unwrap=False,
                               verbose=True,
                               rng=rng,
                               )
    stats = rollout.rollout_stats(rollouts)
    print(f"Rollout stats: {stats}")
    transitions = rollout.flatten_trajectories(rollouts)
    # return 0
    reward_net = BasicRewardNet(eval_env.observation_space, eval_env.action_space, use_state=True, use_action=True, use_next_state=True, hid_sizes=(args.rw_hidden_size,args.rw_hidden_size))
    # print(transitions)

    # Train reward network
    print('-----------------Start training reward network-----------------')
    reward_dataloader = torch.utils.data.DataLoader(RewardDataset(transitions), batch_size=args.rw_batch_size, shuffle=True)
    reward_optimizer = torch.optim.Adam(reward_net.parameters(), lr=args.rw_lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.rw_epochs):
        epoch_loss = 0
        for obs, next_obs, action, done, info in reward_dataloader:
            # print(obs, next_obs, action, done, info)
            reward_optimizer.zero_grad()
            predicted_reward = reward_net(obs, action, next_obs, done)
            
            with torch.no_grad():
                q = iq_agent.infer_q(obs, action)
                q = torch.from_numpy(q).squeeze(-1)
                next_v = agent.infer_v(next_obs).copy()
                y = (1 - done.float()) * iq_agent.gamma * next_v
                reward_target = (q - y)


            loss = criterion(predicted_reward, reward_target.detach())
            loss.backward()
            reward_optimizer.step()
            epoch_loss += loss.item()

        print("RM epoch", epoch, epoch_loss / len(reward_dataloader))
    print('-----------------Reward network training finished-----------------')

    # Wrap the environment with the reward network
    n_envs = 1
    env_train = create_env(env_id, n_envs=n_envs, norm_obs=hyperparams_expert['norm_obs'], norm_reward=False, seed=seed, stats_path=stats_path, env_wrapper=env_wrapper, manual_load=args.test_zoo)
    env_train = RewardVecEnvWrapper(env_train, reward_net.predict_processed)
    print('Environment wrapped with reward network!')

    # Create cqil agent
    # config_path = os.path.join(config_path, algo, env_id+'_1', env_id)
    # hyperparams = get_saved_hyperparams(config_path)
    cqil_trainer = ALGOS[algo](policy='MlpPolicy', env=env_train, verbose=1, seed=seed, 
                                batch_size=args.cqil_batch_size, buffer_size=args.cqil_buffer_size,
                                tau=args.cqil_tau, learning_rate=args.cqil_lr, train_freq=args.cqil_train_freq,
                                gradient_steps=args.cqil_gradient_steps)


    if args.bc_algo == args.algo:
        cqil_trainer.policy = bc_trainer.policy
        print('Initialize with behavior cloning policy!')

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
        save_path = os.path.join(args.save_path, args.env_id, 'cqil')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cqil_trainer.policy.save(os.path.join(save_path, 'cqil_policy_{}_{}_{}_{}_{}_{}_seed_{}.pkl'.format(args.expert_episodes, args.rw_epochs, args.rw_batch_size, args.rw_hidden_size, args.rw_lr, args.iq_episodes, args.seed)))
        print('CQIL Policy saved!')
    if args.save_result is not None:
        result_path = os.path.join(args.save_result, args.env_id, 'cqil')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        np.savez(os.path.join(result_path, 'cqil_result_{}_{}_{}_{}_{}_{}_seed_{}'.format(args.expert_episodes, args.rw_epochs, args.rw_batch_size, args.rw_hidden_size, args.rw_lr, args.iq_episodes, args.seed))
                 , returns=cqil_returns, timesteps=cqil_timesteps, bc_returns=bc_returns, bc_timesteps=bc_timesteps, iq_returns=iq_returns, iq_timesteps=iq_timesteps)
        print('CQIL Result saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument("--env_id", help="Name of environment", default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--algo", default='sac', type=str, help="CQIL agent to use")
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
    parser.add_argument("--bc_epochs", default=100, type=int)
    parser.add_argument("--bc_algo", default='sac', type=str)
    parser.add_argument("--bc_batch_size", default=32, type=int)
    parser.add_argument("--bc_hidden_size", default=256, type=int)
    parser.add_argument("--bc_ent_weight", default=0.00, type=float)
    parser.add_argument("--bc_l2_weight", default=0.01, type=float)

    # Reward Net training parameters
    parser.add_argument("--iq_episodes", default=100, type=int)
    parser.add_argument("--rw_epochs", default=30, type=int)
    parser.add_argument("--rw_batch_size", default=64, type=int)
    parser.add_argument("--rw_lr", default=3e-4, type=float)
    parser.add_argument("--rw_hidden_size", default=256, type=int)

    # IQ learn parameters
    parser.add_argument("--iq_path", default='/home/zli911/imitation/baselines/IQ-Learn/iq_learn/', type=str)

    # cqil parameters
    parser.add_argument("--cqil_timesteps", default=1000000, type=int)
    parser.add_argument("--cqil_batch_size", default=256, type=int)
    parser.add_argument("--cqil_buffer_size", default=300000, type=int)
    parser.add_argument("--cqil_tau", default=0.02, type=int)
    parser.add_argument("--cqil_lr", default=3e-4, type=int)
    parser.add_argument("--cqil_train_freq", default=64, type=int)
    parser.add_argument("--cqil_gradient_steps", default=64, type=int)

    
    args = parser.parse_args()
    print(args)

    train_cqil(args)