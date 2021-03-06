from rlkit.torch.sac.policies import TanhGaussianPolicy
# from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
import numpy as np
from .rl_algorithm import RL_algorithm
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit
import rlkit.torch.pytorch_util as ptu
import torch
import utils

# networks = {individual:, population:}
class SoftActorCritic(RL_algorithm):

    def __init__(self, config, env, replay, networks):
        super().__init__(config, env, replay, networks)

        self._variant_pop = config['rl_algorithm_config']['algo_params_pop']
        self._variant_spec = config['rl_algorithm_config']['algo_params']

        self._ind_qf1 = networks['individual']['qf1']
        self._ind_qf2 = networks['individual']['qf2']
        self._ind_qf1_target = networks['individual']['qf1_target']
        self._ind_qf2_target = networks['individual']['qf2_target']
        self._ind_policy = networks['individual']['policy']

        self._pop_qf1 = networks['population']['qf1']
        self._pop_qf2 = networks['population']['qf2']
        self._pop_qf1_target = networks['population']['qf1_target']
        self._pop_qf2_target = networks['population']['qf2_target']
        self._pop_policy = networks['population']['policy']

        self._batch_size = config['rl_algorithm_config']['batch_size']
        self._nmbr_indiv_updates = config['rl_algorithm_config']['indiv_updates']
        self._nmbr_pop_updates = config['rl_algorithm_config']['pop_updates']

        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            use_automatic_entropy_tuning = False,
            **self._variant_spec
        )

        self._algorithm_pop = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._pop_policy,
            qf1=self._pop_qf1,
            qf2=self._pop_qf2,
            target_qf1=self._pop_qf1_target,
            target_qf2=self._pop_qf2_target,
            use_automatic_entropy_tuning = False,
            **self._variant_pop
        )



    def episode_init(self):
        """ Initializations to be done before the first episode.

        In this case basically creates a fresh instance of SAC for the
        individual networks and copies the values of the target network.
        """
        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            use_automatic_entropy_tuning = False,
            # alt_alpha = self._alt_alpha,
            **self._variant_spec
        )
        if self._config['rl_algorithm_config']['copy_from_global']:
            utils.copy_pop_to_ind(networks_pop=self._networks['population'], networks_ind=self._networks['individual'])


    def single_train_step(self, train_ind=True, train_pop=False):
        if train_ind:
          self._replay.set_mode('species')
          for _ in range(self._nmbr_indiv_updates):
              batch = self._replay.random_batch(self._batch_size)
              self._algorithm_ind.train_from_torch(batch,current_reward,last_reward)

        if train_pop:
          self._replay.set_mode('population')
          for _ in range(self._nmbr_pop_updates):
              batch = self._replay.random_batch(self._batch_size)
              self._algorithm_pop.train(batch)

    @staticmethod
    def create_networks(env, config):
        network_dict = {
            'individual' : SoftActorCritic._create_networks(env=env, config=config),
            'population' : SoftActorCritic._create_networks(env=env, config=config),
        }
        return network_dict

    @staticmethod
    def _create_networks(env, config):
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        net_size = config['rl_algorithm_config']['net_size']
        hidden_sizes = [net_size] * config['rl_algorithm_config']['network_depth']
        # hidden_sizes = [net_size, net_size, net_size]
        qf1 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf1_target = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2_target = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        policy = TanhGaussianPolicy(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        ).to(device=ptu.device)

        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {'qf1' : qf1, 'qf2' : qf2, 'qf1_target' : qf1_target, 'qf2_target' : qf2_target, 'policy' : policy}

    @staticmethod
    def get_q_network(networks):
        return networks['qf1']

    @staticmethod
    def get_policy_network(networks):
        return networks['policy']
