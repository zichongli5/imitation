import gymnasium as gym
import pybullet_envs_gymnasium
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from wrappers.atari_wrapper import ScaledFloatFrame, FrameStack, FrameStackEager, PyTorchFrame
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions

import envs
import numpy as np
import os
import pickle

# Register all custom envs
envs.register_custom_envs()

def make_dcm(cfg):
    import dmc2gym
    """Helper function to create dm_control environment"""
    if cfg.env.name == 'dmc_ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env.name == 'dmc_point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.name.split('_')[1]
        task_name = '_'.join(cfg.env.name.split('_')[2:])
    
    if cfg.env.from_pixels:
        # Set env variables for Mujoco rendering
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

        # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
        camera_id = 2 if domain_name == 'quadruped' else 0

        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=False,
                        from_pixels=True,
                        height=cfg.env.image_size,
                        width=cfg.env.image_size,
                        frame_skip=cfg.env.action_repeat,
                        camera_id=camera_id)

        print(env.observation_space.dtype)
        # env = FrameStack(env, k=cfg.env.frame_stack)
        env = FrameStackEager(env, k=cfg.env.frame_stack)
        
    else:
        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def make_atari(env):
    env = AtariWrapper(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env

def is_atari(env_name):
    return env_name in ['PongNoFrameskip-v4', 
                        'BreakoutNoFrameskip-v4', 
                        'SpaceInvadersNoFrameskip-v4', 
                        'BeamRiderNoFrameskip-v4',
                        'QbertNoFrameskip-v4',
                        'SeaquestNoFrameskip-v4']


def make_env(args, monitor=True):
    if 'dmc' in args.env.name:
        env = make_dcm(args)
    else:
        env = gym.make(args.env.name)
    
    if monitor:
        env = Monitor(env, "gym")

    if is_atari(args.env.name):
        env = make_atari(env)

    # Normalize box actions to [-1, 1]
    env = check_and_normalize_box_actions(env)
    return env

def create_env(env_id, n_envs=1, norm_obs=True, norm_reward=False, norm_action=False, stats_path=None, seed=0, env_wrapper=None, manual_load=False):
    # env = gym.make(env_id, apply_api_compatibility=True)
    # env = DummyVecEnv([lambda: env])
    # env.seed(seed=0)
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
    
    # env.norm_reward = norm_reward
    # env.norm_obs = norm_obs

    if norm_action:
        raise NotImplementedError
    
    return env

def load_stats_to_env(stats_path, env):
    '''
    Load stats to normalization; This function can keep the Gymnasium natrure of env
    '''
    with open(stats_path, 'rb') as file:
        saved_stats = pickle.load(file)
    # Assign the loaded normalization statistics
    env.obs_rms.mean = saved_stats.obs_rms.mean
    env.obs_rms.var = saved_stats.obs_rms.var
    env.obs_rms.count = saved_stats.obs_rms.count
    env.ret_rms.mean = saved_stats.ret_rms.mean
    env.ret_rms.var = saved_stats.ret_rms.var
    env.ret_rms.count = saved_stats.ret_rms.count
    env.training = False

    return env