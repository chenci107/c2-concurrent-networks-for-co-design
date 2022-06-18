import numpy as np
from bayes_opt import BayesianOptimization
import rlkit.torch.pytorch_util as ptu
import numpy as np
import torch

class BO_batch():
    def __init__(self,config,replay,env):
        self._config = config
        self._replay = replay
        self._env = env
        if 'state_batch_size' in self._config.keys():
            self._state_batch_size = self._config['state_batch_size']  # 32,Size of the batch used during the design optimization process to estimate fitness of design
        else:
            self._state_batch_size = 32

        self._init_points = self._config['bo_params']['init_points']
        self._n_iter = self._config['bo_params']['n_iter']

    def optimize_design(self,design,q_network,policy_network):
        self._replay.set_mode('start')
        initial_state = self._replay.random_batch(self._state_batch_size)
        initial_state = initial_state['observations']  # numpy: [batch_size * (original_state + design_parameters)]
        design_idxs = self._env.get_design_dimensions()


        def get_Q_from_design(design):
            state_batch = initial_state.copy()
            state_batch[:,design_idxs] = design
            network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
            # action = pop_network.get_batch_action(state_batch)
            # output = pop_network.get_batch_q_value(state_batch,action)
            if self._config['rl_method'] == 'SoftActorCritic':
                action, _, _, _, _, _, _, _, = policy_network(network_input,
                                                              deterministic=True)  # action: (batch_size, action_dim)
                #  action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value, policies.py
            elif self._config['rl_method'] == 'TD3':
                action = policy_network(network_input)
            else:
                raise ValueError

            output = q_network(network_input,action)
            loss = -output.mean().sum()
            fval = float(loss.item())

            return fval

        def f_qval(des_1,des_2,des_3,des_4,des_5,des_6):
            reward = get_Q_from_design(np.array([des_1,des_2,des_3,des_4,des_5,des_6]))
            return reward

        bounds = {'des_1': (0.5, 1.5), 'des_2': (0.5, 1.5), 'des_3': (0.5, 1.5), 'des_4': (0.5, 1.5),
                  'des_5': (0.5, 1.5), 'des_6': (0.8, 1.5)}

        optimizer = BayesianOptimization(f=f_qval, pbounds=bounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=self._init_points, n_iter=self._n_iter)

        new_design = optimizer.max['params']
        new_design_res = []
        for key, value in new_design.items():
            new_design_res.append(value)

        print('new_design', new_design_res)

        return new_design_res




