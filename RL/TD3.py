import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import TanhMlpPolicy
from rlkit.torch.networks import FlattenMlp
from .td3bc_rlkit import TD3BCTrainer as TD3BC_rlkit
from .td3_adapt_rlkit import TD3Trainer as TD3Adapt_rlkit
from .rl_algorithm import RL_algorithm
from rlkit.torch.core import np_to_pytorch_batch
import numpy as np
import torch
import utils


class TD3(RL_algorithm):
    def __init__(self,config,env,replay,networks):
        super().__init__(config,env,replay,networks)
        self._variant_pop = config['rl_algorithm_config']['algo_params_pop']
        self._variant_spec = config['rl_algorithm_config']['algo_params_adapt']

        self._ind_qf1 = networks['individual']['qf1']
        self._ind_qf2 = networks['individual']['qf2']
        self._ind_qf1_target = networks['individual']['qf1_target']
        self._ind_qf2_target = networks['individual']['qf2_target']
        self._ind_policy = networks['individual']['policy']
        self._ind_policy_target = networks['individual']['policy_target']

        self._pop_qf1 = networks['population']['qf1']
        self._pop_qf2 = networks['population']['qf2']
        self._pop_qf1_target = networks['population']['qf1_target']
        self._pop_qf2_target = networks['population']['qf2_target']
        self._pop_policy = networks['population']['policy']
        self._pop_policy_target = networks['population']['policy_target']

        self._batch_size = config['rl_algorithm_config']['batch_size']
        self._nmbr_indiv_updates = config['rl_algorithm_config']['indiv_updates']
        self._nmbr_pop_updates = config['rl_algorithm_config']['pop_updates']

        self._algorithm_ind = TD3Adapt_rlkit(
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            target_policy=self._ind_policy_target,
            **self._variant_spec)

        self._algorithm_pop = TD3BC_rlkit(
            policy=self._pop_policy,
            qf1=self._pop_qf1,
            qf2=self._pop_qf2,
            target_qf1=self._pop_qf1_target,
            target_qf2=self._pop_qf2_target,
            target_policy=self._pop_policy_target,
            **self._variant_pop)

    def episode_init(self):
        self._algorithm_ind = TD3Adapt_rlkit(
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            target_policy=self._ind_policy_target,
            **self._variant_spec)

        if self._config['rl_algorithm_config']['copy_from_global']:
            utils.copy_pop_to_ind(networks_pop=self._networks['population'],networks_ind=self._networks['individual'])

    def single_train_step(self,train_ind=True,train_pop=False,current_reward=None,last_reward=None):
        print('Training policy...')
        if train_ind:
            self._replay.set_mode("species")
            for _ in range(self._nmbr_indiv_updates):
                batch = np_to_pytorch_batch(self._replay.random_batch(self._batch_size))
                self._algorithm_ind.train_from_torch(batch, current_reward, last_reward)

        if train_pop:
            self._replay.set_mode("population")
            for _ in range(self._nmbr_pop_updates):
                batch = np_to_pytorch_batch(self._replay.random_batch(self._batch_size))
                self._algorithm_pop.train_from_torch(batch)

    @staticmethod
    def create_networks(env,config):
        network_dict = {
            'individual': TD3._create_networks(env=env,config=config),
            'population': TD3._create_networks(env=env,config=config),}

        return network_dict


    @staticmethod
    def _create_networks(env,config):
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        net_size = config['rl_algorithm_config']['net_size']
        hidden_sizes = [net_size] * config['rl_algorithm_config']['network_depth']
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

        target_qf1 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim+action_dim,
            output_size=1
        ).to(device=ptu.device)

        target_qf2 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1
       ).to(device=ptu.device)

        policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
        ).to(device=ptu.device)

        target_policy = TanhMlpPolicy(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes
        ).to(ptu.device)

        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value,clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {'qf1': qf1, 'qf2': qf2, 'qf1_target': target_qf1, 'qf2_target':target_qf2,
                'policy': policy, 'policy_target':target_policy}

    @staticmethod
    def get_q_network(networks):
        return networks['qf1']

    @staticmethod
    def get_policy_network(networks):
        return networks['policy']




