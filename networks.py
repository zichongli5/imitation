from stable_baselines3.common import policies, torch_layers
from imitation.rewards.reward_nets import RewardNet
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import DiagGaussianDistribution, SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, create_mlp
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy

import copy

import torch.nn as nn
import torch
import math
import numpy as np

class RewardNet_from_policy(RewardNet):
    def __init__(self, policy, alpha=1.0):
        super().__init__(observation_space=policy.observation_space, action_space=policy.action_space)
        self.policy = policy
        self.policy.set_training_mode(False)
        self.alpha = alpha
        # action_space_low = policy.action_space.low
        # action_space_high = policy.action_space.high

        # action_range = torch.tensor(action_space_high - action_space_low)
        action_num = get_action_dim(policy.action_space)
        print('action_num:', action_num, self.policy.action_dist)
        self.log_prior = -math.log(action_num)

    
    def forward(self, state, action, next_state, done):
        print(action)
        '''
        Compute reward from policy
        '''
        with torch.no_grad():
            _, log_prob, _ = self.policy.evaluate_actions(state, action)
            reward = self.alpha * (log_prob - self.log_prior)
            print(log_prob)
        # print(self.log_prior,reward)
            print(reward)
        return reward

    def get_logits(self, state, action, next_state, done):
        with torch.no_grad():
            logits, _, _ = self.policy.evaluate_actions(state, action)
            logits_act = logits.gather(1, action.long().unsqueeze(1)).squeeze(1)
        # print(self.log_prior,reward)
        return logits_act
    

    # def predict_processed(self, state, action, next_state, done, **kwargs):
    #     '''
    #     Compute reward from policy
    #     '''
    #     return self(state, action, next_state, done)

 
class SACBCPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        squash = True
        self.squash = squash
        action_dim = get_action_dim(self.action_space)
        if not squash:
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
        # log_std = torch.zeros(log_std.size()) - 3
        distribution = self.actor.action_dist.proba_distribution(mean_actions, log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        if entropy is None:
            entropy = -log_prob.mean()
        return mean_actions, log_prob, entropy


class PPOBCPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,share_features_extractor=False)


    def evaluate_actions(self, obs, actions):
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        mean_actions = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return mean_actions, log_prob, entropy

class PPOBCCNNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,share_features_extractor=False)


    def evaluate_actions(self, obs, actions):
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pi_features, vf_features = features
        latent_pi = self.mlp_extractor.forward_actor(pi_features)
        mean_actions = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        log_prob = distribution.log_prob(actions.to(self.device))
        entropy = distribution.entropy()
        return mean_actions, log_prob, entropy
    

from imitation.algorithms.bc import BC
from imitation.algorithms.bc import BatchIteratorWithEpochEndCallback, enumerate_batches, RolloutStatsComputer
from imitation.data import rollout, types
from torch.utils.data import DataLoader, RandomSampler
from imitation.util import util
import dataclasses
import tqdm

@dataclasses.dataclass(frozen=True)
class BC_ATMetric:
    """Container for the different components of behavior cloning loss."""

    neglogp: torch.Tensor
    ent_loss: torch.Tensor  # set to 0 if entropy is None
    l2_loss: torch.Tensor
    loss: torch.Tensor

def tau_collate_fn(batch):
    batch_acts_and_dones = [
        {k: np.array(v) for k, v in sample.items() if k in ["acts", "dones"]}
        for sample, _ in batch
    ]

    result = torch.utils.data.dataloader.default_collate(batch_acts_and_dones)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample, _ in batch]
    result["obs"] = types.stack_maybe_dictobs([sample["obs"] for sample, _ in batch])
    result["next_obs"] = types.stack_maybe_dictobs([sample["next_obs"] for sample, _ in batch])
    result["index"] = torch.tensor([key for _, key in batch])
    return result



def dataclass_quick_asdict(obj):
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d

def getitem(self, key):
    """See TransitionsMinimal docstring for indexing and slicing semantics."""
    d = dataclass_quick_asdict(self)
    d_item = {k: v[key] for k, v in d.items()}

    if isinstance(key, slice):
        # Return type is the same as this dataclass. Replace field value with
        # slices.
        return dataclasses.replace(self, **d_item)
    else:
        assert isinstance(key, int)
        return d_item, key


class wrap_traisition_data:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, key):
        return getitem(self.data, key)

    def __len__(self):
        return len(self.data)


class BC_AT(BC):
    def __init__(self, *args, N, tau_init, tau_min, tau_max, rho, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base BC class
        self.N = len(kwargs['demonstrations'])
        self.ent_weight = kwargs['ent_weight']
        self.l2_weight = kwargs['l2_weight']
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.rho = rho
        # self.tau = torch.ones(self.N) * self.tau_init
        # initialize tau as parameters
        # self.tau = nn.Parameter(torch.ones(self.N) * self.tau_init)
        self.tau = torch.ones(self.N) * self.tau_init
        self.tau = self.tau.to(self.policy.device)
        # list that stores the tau values along the training
        self.tau_list = []
        # self.optimizer.add_param_group({'params': [self.tau], 'lr': 0.01})
        transition_data = kwargs['demonstrations']
        transition_data = wrap_traisition_data(transition_data)
        self._demo_data_loader = torch.utils.data.DataLoader(transition_data, batch_size=self.minibatch_size,  collate_fn=tau_collate_fn, shuffle=True, drop_last=True)
        # self.loss_calculator = BC_ATLoss(ent_weight=self.ent_weight, l2_weight=self.l2_weight)
    
    def cal_loss(self, policy, obs, acts, index, batch_num):
        tau = self.tau[index]
        # tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)
        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        acts = util.safe_to_tensor(acts)

        (logits, log_prob, entropy) = policy.evaluate_actions(tensor_obs, acts)
        # print('logits:', logits[0])


        def compute_gradient_hessian_tau(logits, tau, rho, M):
            softmax = torch.softmax(logits / tau.unsqueeze(-1), dim=1)
            f_x_tau = logits / tau.unsqueeze(-1)
            log_sum_exp = torch.logsumexp(logits / tau.unsqueeze(-1), dim=1)

            term1 = rho - torch.log(torch.tensor(M))
            term2 = log_sum_exp
            term3 = -(1/tau) * torch.sum(softmax * logits, dim=1)/torch.sum(softmax, dim=1)

            gradient = term1 + term2 + term3

            sum_exp = torch.sum(softmax, dim=1)
            sum_exp_fx = torch.sum(softmax * logits, dim=1)
            sum_exp_fx2 = torch.sum(softmax * logits**2, dim=1)

            term1 = -(1 / tau**3) * (sum_exp_fx ** 2) / (sum_exp ** 2)
            term2 = (1 / tau**3) * sum_exp_fx2 / sum_exp

            hessian = term1 + term2
            return gradient, hessian


        # logits is the logits of the action distribution
        # compute log prob from the logits with temperature tau
        # logits: (bsz, num_actions)
        # acts: (bsz, )
        # tau: (bsz, )
        # update tau using newton method, n iterations
        if batch_num>100:
            with torch.no_grad():
                for i in range(6):
                    gradient_tau, hessian_tau = compute_gradient_hessian_tau(logits, tau, self.rho, M=logits.size(1))
                    tau = tau - gradient_tau / (hessian_tau + 1e-6)
                    tau = torch.clamp(tau, min=self.tau_min, max=self.tau_max)
        # else:
        #     print(batch_num)
        
        self.tau[index] = tau
        f_x_tau = logits / tau.unsqueeze(-1)
        log_sum_exp = torch.logsumexp(logits / tau.unsqueeze(-1), dim=1)
        log_prob_tau = f_x_tau - log_sum_exp.unsqueeze(-1)
        log_prob_tau = log_prob_tau.gather(1, acts.long().unsqueeze(1)) * tau
        # count correct actions
        # correct = torch.sum(torch.argmax(logits, dim=1) == acts.long())
        # compute the entropy of the action distribution with temperature tau
        # entropy_tau = -torch.sum(torch.exp(log_prob_tau) * log_prob_tau, dim=-1)

        # prob_true_act = torch.exp(log_prob).mean()
        log_prob_tau = log_prob_tau.mean()
        entropy_tau = entropy.mean()

        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * entropy_tau
        neglogp = -log_prob_tau
        l2_loss = self.l2_weight * l2_norm
        tau_reg = (self.rho - math.log(logits.size(1))) * tau.mean()
        loss = neglogp + ent_loss + l2_loss + tau_reg


        return BC_ATMetric(
            neglogp=neglogp,
            ent_loss=ent_loss,
            l2_loss=l2_loss,
            loss=loss,
        ), (logits.gather(1, acts.long().unsqueeze(1))).squeeze(1), tau

    def train(
        self,
        *,
        n_epochs = None,
        n_batches = None,
        on_epoch_end = None,
        on_batch_end = None,
        log_interval = 500,
        progress_bar = True,
        reset_tensorboard = False,
    ):

        if reset_tensorboard:
            self._bc_logger.reset_tensorboard_steps()
        self._bc_logger.log_epoch(0)

        compute_rollout_stats = RolloutStatsComputer(
            None,
            5,
        )

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(
                    f"Epoch {epoch_number} {total_num_epochs_str}",
                    pos=1,
                )
            self._bc_logger.log_epoch(epoch_number + 1)
            if on_epoch_end is not None:
                on_epoch_end()

        mini_per_batch = self.batch_size // self.minibatch_size
        n_minibatches = n_batches * mini_per_batch if n_batches is not None else None

        assert self._demo_data_loader is not None
        demonstration_batches = BatchIteratorWithEpochEndCallback(
            self._demo_data_loader,
            n_epochs,
            n_minibatches,
            _on_epoch_end,
        )
        batches_with_stats = enumerate_batches(demonstration_batches)
        tqdm_progress_bar = None

        if progress_bar:
            batches_with_stats = tqdm.tqdm(
                batches_with_stats,
                unit="batch",
                total=n_minibatches,
            )
            tqdm_progress_bar = batches_with_stats

        def process_batch():
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_num % log_interval == 0:
                rollout_stats = compute_rollout_stats(self.policy, self.rng)

                self._bc_logger.log_batch(
                    batch_num,
                    minibatch_size,
                    num_samples_so_far,
                    training_metrics,
                    rollout_stats,
                )

            if on_batch_end is not None:
                on_batch_end()

        self.optimizer.zero_grad()
        for (
            batch_num,
            minibatch_size,
            num_samples_so_far,
        ), batch in batches_with_stats:
            # unwraps the observation if it's a dictobs and converts arrays to tensors
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["obs"]),
            )
            acts = util.safe_to_tensor(batch["acts"], device=self.policy.device).to(self.policy.device)
            training_metrics, act_logits, tau = self.cal_loss(self.policy, obs_tensor, acts, batch['index'], batch_num)
            if batch_num % 125 == 0:
                self.tau_list.append(self.tau.unsqueeze(0).clone().cpu())

            # Renormalise the loss to be averaged over the whole
            # batch size instead of the minibatch size.
            # If there is an incomplete batch, its gradients will be
            # smaller, which may be helpful for stability.
            loss = training_metrics.loss * minibatch_size / self.batch_size
            loss.backward()

            batch_num = batch_num * self.minibatch_size // self.batch_size
            if num_samples_so_far % self.batch_size == 0:
                process_batch()
        if num_samples_so_far % self.batch_size != 0:
            # if there remains an incomplete batch
            batch_num += 1
            process_batch()


class BC_base(BasePolicy):
    '''
    Basic BC policy, used to test the code, get similar results with SACBCPolicy
    '''

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        lr_schedule,
        features_extractor = FlattenExtractor,
        activation_fn = nn.ReLU,
        normalize_images = True,
        squash_output = True
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=squash_output,
        )

        # Save arguments to re-create object at loading
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(self.features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else self.features_dim
        if squash_output:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        else:
            self.action_dist = DiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def get_action_dist_params(self, obs):
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, -20, 2)
        return mean_actions, log_std, {}

    def evaluate_actions(self, obs, actions):

        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        distribution = self.action_dist.proba_distribution(mean_actions, log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        if entropy is None:
            entropy = -log_prob.mean()

        return None, log_prob, entropy

    def _predict(self, obs, deterministic= False):
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)



