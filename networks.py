from stable_baselines3.common import policies, torch_layers
from imitation.rewards.reward_nets import RewardNet
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution

import copy

import torch

class RewardNet_from_policy(RewardNet):
    def __init__(self, policy, alpha=1.0):
        super().__init__(observation_space=policy.observation_space, action_space=policy.action_space)
        self.policy = copy.deepcopy(policy)
        self.policy.set_training_mode(False)
        self.alpha = alpha
        action_space_low = policy.action_space.low
        action_space_high = policy.action_space.high

        action_range = torch.tensor(action_space_high - action_space_low)
        self.log_prior = -torch.log(action_range).sum()

    
    def forward(self, state, action, next_state, done):
        '''
        Compute reward from policy
        '''
        with torch.no_grad():
            _, log_prob, _ = self.policy.evaluate_actions(state, action)
            reward = self.alpha * (log_prob - self.log_prior)
        # print(self.log_prior,reward)
        return reward
    

    # def predict_processed(self, state, action, next_state, done, **kwargs):
    #     '''
    #     Compute reward from policy
    #     '''
    #     return self(state, action, next_state, done)

 
class SACBCPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        action_dim = get_action_dim(self.action_space)
        self.actor.action_dist = DiagGaussianDistribution(action_dim)
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs)
        distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        # print(distribution)
        # print(mean_actions[0], log_std[0], log_prob[0], entropy[0])
        # import sys
        # sys.exit()
        return None, log_prob, entropy

