import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from .design_optimization import Design_Optimization

class PSO_batch(Design_Optimization):

    def __init__(self, config, replay, env):
        self._config = config
        self._replay = replay
        self._env = env

        if 'state_batch_size' in self._config.keys():
            self._state_batch_size = self._config['state_batch_size']
        else:
            self._state_batch_size = 32

        self._n_particles = self._config['pso_params']['n_particles']
        self._n_iters = self._config['pso_params']['iters']


    def optimize_design(self, design, q_network, policy_network):
        self._replay.set_mode('start')
        initial_state = self._replay.random_batch(self._state_batch_size)
        initial_state = initial_state['observations']
        design_idxs = self._env.get_design_dimensions()
        # initial_state = initial_state[:,:-len(design)]
        # state_tensor = torch.from_numpy(initial_state).to(device=ptu.device, dtype=torch.float32)

        # initial_design = np.array(self._current_design)
        # initial_design = np.array(design)

        def f_qval(x_input, **kwargs):
            shape = x_input.shape  # (n_particles, dimensions) (700,4)
            cost = np.zeros((shape[0],))   # (n_particles,) (700,)
            with torch.no_grad():
                for i in range(shape[0]):
                    x = x_input[i:i+1,:]  # (dimensions,) (4,)
                    state_batch = initial_state.copy() # (batch_size, (original_state + design_params)) (32,13+4)
                    state_batch[:,design_idxs] = x     # replace the original design parameters with the design parameters after PSO, (btach_size, (original_state + design_params))
                    network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)

                    if self._config['rl_method'] == 'SoftActorCritic':
                        action, _, _, _, _, _, _, _, = policy_network(network_input, deterministic=True)  # action: (batch_size, action_dim)
                        #  action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value, policies.py
                    elif self._config['rl_method'] == 'TD3':
                        action = policy_network(network_input)
                    else:
                        raise ValueError

                    output = q_network(network_input, action)
                    loss = -output.mean().sum()
                    fval = float(loss.item())
                    cost[i] = fval
            return cost

        lower_bounds = [l for l, _ in self._env.design_params_bounds]
        lower_bounds = np.array(lower_bounds)
        upper_bounds = [u for _, u in self._env.design_params_bounds]
        upper_bounds = np.array(upper_bounds)
        bounds = (lower_bounds, upper_bounds)
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        #options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
        optimizer = ps.single.GlobalBestPSO(n_particles=self._n_particles, dimensions=len(design), bounds=bounds, options=options)

        # Perform optimization
        cost, new_design = optimizer.optimize(f_qval, print_step=100, iters=self._n_iters, verbose=3) #, n_processes=2)
        return new_design
