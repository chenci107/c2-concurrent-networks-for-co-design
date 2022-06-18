from bayes_opt import BayesianOptimization
import numpy as np
import torch

class BO_simulation():
    def __init__(self,config,replay,env):
        self._config = config
        self._replay = replay
        self._env = env
        self._episode_length = 1000
        self._init_points = self._config['bo_params']['init_points']
        self._n_iter = self._config['bo_params']['n_iter']

    def optimize_design(self,design,q_network,policy_network):

        def get_reward_form_design(design):
            self._env.set_new_design(design)
            state = self._env.reset()
            reward_episode = []
            done = False
            number_of_steps = 0
            while not (done) and number_of_steps <= self._episode_length:
                number_of_steps += 1
                if self._config['rl_method'] == 'SoftActorCritic':
                    action, _ = policy_network.get_action(state, deterministic=True)
                elif self._config['rl_method'] == 'TD3':
                    action, _ = policy_network.get_action(state)
                else:
                    raise ValueError
                new_state, reward, done, info = self._env.step(action)
                reward_episode.append(float(reward))
                state = new_state
            reward_mean = np.mean(reward_episode)
            return reward_mean

        def f_qval(des_1,des_2,des_3,des_4,des_5,des_6):
            reward = get_reward_form_design([des_1,des_2,des_3,des_4,des_5,des_6])
            return reward

        bounds = {'des_1':(0.5,1.5),'des_2':(0.5,1.5),'des_3':(0.5,1.5),'des_4':(0.5,1.5),'des_5':(0.5,1.5),'des_6':(0.5,1.5)}

        optimizer = BayesianOptimization(f=f_qval, pbounds=bounds, verbose=2, random_state=1)
        optimizer.maximize(init_points=self._init_points, n_iter=self._n_iter)

        new_design = optimizer.max['params']
        new_design_res = []
        for key, value in new_design.items():
            new_design_res.append(value)

        print('new_design', new_design_res)

        return new_design_res
