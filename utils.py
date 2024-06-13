import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.utils import set_random_seed, constant_fn
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy


# Patch and register pybullet envs
import rl_zoo3.gym_patches
import pybullet_envs
import pybullet_envs_gymnasium

# import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
from rl_zoo3 import ALGOS
from rl_zoo3.utils import StoreDict, get_model_path, linear_schedule, get_callback_list, get_class_by_name, get_latest_run_id, get_wrapper_class


import numpy as np
import os
import tqdm
import yaml
import torch.nn as nn
import torch
import copy
import pickle
def is_atari(env_name):
    return env_name in ['PongNoFrameskip-v4', 
                        'BreakoutNoFrameskip-v4', 
                        'SpaceInvadersNoFrameskip-v4', 
                        'BeamRiderNoFrameskip-v4',
                        'QbertNoFrameskip-v4',
                        'SeaquestNoFrameskip-v4',
                        'AsteroidsNoFrameskip-v4']

def create_env(env_id, n_envs=1, norm_obs=True, norm_reward=False, norm_action=False, stats_path=None, seed=0, env_wrapper=None, frame_num=1, manual_load=False, reward_wrap=None):
    # env = gym.make(env_id, apply_api_compatibility=True)
    # env = DummyVecEnv([lambda: env])
    # env.seed(seed=0)
    if is_atari(env_id):
        env = make_atari_env(env_id, n_envs, seed=seed)
        env = VecFrameStack(env, n_stack=frame_num)
        env = VecTransposeImage(env)
    else:
        env = make_vec_env(env_id, n_envs, seed=seed, wrapper_class=env_wrapper)


    if norm_obs or norm_reward:
        if stats_path is not None:
            if not manual_load:
                env = VecNormalize.load(stats_path, env)
            else:
                print('Load Stats manually!')
                env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
                env = load_stats_to_env(stats_path, env)
            print('Stats loaded for normalization!')
            env.training = False
        else:
            print('No stats path provided! Initialize the stats for normalization!')
            env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)
            env.training = True
            
    if reward_wrap is not None:
        env = RewardVecEnvWrapper(env, reward_wrap.predict_processed, 0)
    # env.norm_reward = norm_reward
    # env.norm_obs = norm_obs

    if norm_action:
        raise NotImplementedError
    
    return env


def evaluate(env, model, num_episodes=10, deterministic=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False
        total_r = 0
        total_step = 0

        while not done:
            action, _ = model.predict(state, deterministic=deterministic)
            # action = np.clip(action, model.action_space.low, model.action_space.high)  # type: ignore[assignment, arg-type]
            next_state, reward, done, info = env.step(action)
            print('eval:', action, reward)
            state = next_state
            total_r += reward
            total_step += 1
            # if 'episode' in info.keys():
            #     total_returns.append(info['episode']['r'])
            #     total_timesteps.append(info['episode']['l'])
        total_returns.append(total_r)
        total_timesteps.append(total_step)

    return total_returns, total_timesteps


def load_pretrained_expert(env, algo, model_path=None, eval=False, eval_steps=1000):
    '''
    Load expert model from model_path and evaluate it on the environment
    '''
    assert model_path is not None, 'No expert path provided?!'
    model = ALGOS[algo].load(model_path)
    print('-------------- Policy Architecture --------------')
    print(model.policy)
    _ = env.reset()

    if eval:
        # Evaluate the trained agent
        print('Evaluate the trained agent')
        total_returns, total_timesteps = evaluate_policy(model, env,  n_eval_episodes=eval_steps)
        print('Return: {} +/- {}'.format(np.mean(total_returns), np.std(total_returns)))
        print('Timesteps: {} +/- {}'.format(np.mean(total_timesteps), np.std(total_timesteps)))

    return model


def get_saved_hyperparams(config_path):
    """
    Retrieve saved hyperparameters given a path.
    Return empty dict if the path is not valid.

    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams = {}
    if not os.path.isdir(config_path):
        return hyperparams
    else:
        config_file = os.path.join(config_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(config_path, "config.yml")) as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        # else:
        #     obs_rms_path = os.path.join(config_path, "obs_rms.pkl")
        #     hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                hyperparams["norm_obs"] = normalize_kwargs['norm_obs']
                hyperparams["norm_reward"] = normalize_kwargs['norm_reward']
            else:
                hyperparams["norm_obs"] = hyperparams["normalize"]
                hyperparams["norm_reward"] = hyperparams["normalize"]
        else:
            hyperparams["norm_obs"] = False
            hyperparams["norm_reward"] = False
        if 'frame_stack' not in hyperparams.keys():
            hyperparams['frame_stack'] = 1
    return hyperparams


def preprocess_schedules(hyperparams):
    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
    return hyperparams

def preprocess_hyperparams(hyperparams):
    '''
    Delete unnecessary hyperparameters
    '''
    hyperparams = copy.deepcopy(hyperparams)
    n_envs = hyperparams.get("n_envs", 1)
    print(f"Using {n_envs} environments")

    # Convert schedule strings to objects
    hyperparams = preprocess_schedules(hyperparams)

    # Pre-process train_freq
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    # Pre-process policy/buffer keyword arguments
    # Convert to python object if needed
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    # Delete keys so the dict can be pass to the model constructor
    for key in ["policy_kwargs", "n_envs", "n_timesteps", "frame_stack", "env_wrapper", "normalize", "normalize_kwargs", "norm_obs", "norm_reward","seed"]:
        if key in hyperparams.keys():
            del hyperparams[key]

    # import the policy when using a custom policy
    if "policy" in hyperparams and "." in hyperparams["policy"]:
        hyperparams["policy"] = get_class_by_name(hyperparams["policy"])
    
    return hyperparams

def load_stats_to_env(stats_path, env):
    '''
    Load stats to normalization; This function can keep the Gymnasium natrure of env
    '''
    with open(stats_path, 'rb') as file:
        saved_stats = pickle.load(file)
    # Assign the loaded normalization statistics
    if env.norm_obs:
        env.obs_rms.mean = saved_stats.obs_rms.mean
        env.obs_rms.var = saved_stats.obs_rms.var
        env.obs_rms.count = saved_stats.obs_rms.count
    if env.norm_reward:
        env.ret_rms.mean = saved_stats.ret_rms.mean
        env.ret_rms.var = saved_stats.ret_rms.var
        env.ret_rms.count = saved_stats.ret_rms.count
    env.training = False

    return env


def compat_forward(self, obs, deterministic=False):
        return self._predict(obs, deterministic=deterministic)

def action_log_prob(self, obs):
    distribution = self.get_distribution(obs)
    action = distribution.get_actions(deterministic=False)
    log_prob = distribution.log_prob(action)
    return action, log_prob

 
def wrap_bc_policy(policy):
    policy.forward = compat_forward.__get__(policy)
    policy.action_log_prob = action_log_prob.__get__(policy)
    return policy

def predict(self, obs, state=None, done=None, deterministic=True):
    return self.choose_action(obs, sample=(not deterministic)), None

def wrap_iq_agent(iq_agent):
    iq_agent.predict = predict.__get__(iq_agent)
    return iq_agent

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, trainsitions):
        self.transitions = copy.deepcopy(trainsitions)
    
    def __getitem__(self, index):
        return self.transitions.obs[index], self.transitions.next_obs[index], self.transitions.acts[index], self.transitions.dones[index], self.transitions.infos[index], self.transitions.rews[index]
    
    def __len__(self):
        return len(self.transitions)

def pearson_correlation(tensor1, tensor2):
    """ Compute Pearson Correlation Coefficient for two PyTorch tensors. """
    mean1 = torch.mean(tensor1)
    mean2 = torch.mean(tensor2)
    numerator = torch.sum((tensor1 - mean1) * (tensor2 - mean2))
    denominator = torch.sqrt(torch.sum((tensor1 - mean1) ** 2) * torch.sum((tensor2 - mean2) ** 2))
    return numerator / denominator