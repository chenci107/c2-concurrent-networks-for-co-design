from RL.soft_actor import SoftActorCritic
from RL.TD3 import TD3
from DO.pso_batch import PSO_batch
from DO.pso_sim import PSO_simulation
from DO.bo_batch import BO_batch
from DO.bo_sim import BO_simulation
from Environments import evoenvs as evoenvs
from Environments import evoenvs_halfcheetah
import utils
import time
from RL.evoreplay import EvoReplayLocalGlobalStart
import numpy as np
import os
import csv
import torch


def select_design_opt_alg(alg_name):
    if alg_name == "pso_batch":
        return PSO_batch
    elif alg_name == "pso_sim":
        return PSO_simulation
    elif alg_name == "bo_batch":
        return BO_batch
    elif alg_name == 'bo_sim':
        return BO_simulation
    else:
        raise ValueError("Design Optimization method not found.")

def select_environment(env_name):
    if env_name == 'HalfCheetah':
        return evoenvs.HalfCheetahEnv
    elif env_name == 'HalfCheetah_mujoco':
        return evoenvs_halfcheetah.HalfCheetahEnv
    else:
        raise ValueError("Environment class not found.")

def select_rl_alg(rl_name):
    if rl_name == 'SoftActorCritic':
        return SoftActorCritic
    elif rl_name == 'TD3':
        return TD3
    else:
        raise ValueError('RL method not fund.')

class Coadaptation(object):
    def __init__(self, config):

        self._config = config
        utils.move_to_cuda(self._config)
        self._episode_length = self._config['steps_per_episodes']
        self._reward_scale = 1.0
        '''Initialize environment'''
        self._env_class = select_environment(self._config['env']['env_name'])
        self._env = self._env_class(config=self._config)
        self.seed = self._config['seed']
        self._env.seed(self.seed)
        '''Initialize replay buffer'''
        self._replay = EvoReplayLocalGlobalStart(self._env,
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7))
        '''Reinforcement Learning'''
        self._rl_alg_class = select_rl_alg(self._config['rl_method'])
        self._networks = self._rl_alg_class.create_networks(env=self._env, config=config)
        self._rl_alg = self._rl_alg_class(config=self._config, env=self._env , replay=self._replay, networks=self._networks)
        '''Optimization method'''
        self._do_alg_class = select_design_opt_alg(self._config['design_optim_method'])
        self._do_alg = self._do_alg_class(config=self._config, replay=self._replay, env=self._env)
        utils.move_to_cuda(self._config)

        self._last_single_iteration_time = 0
        self._design_counter = 0
        self._episode_counter = 0
        self._data_design_type = 'Initial'

        ### add for td3 ###
        self.max_action = self._env.action_space.high[0]  # 1
        self.expl_noise = 0.1
        self.action_dim = int(np.prod(self._env.action_space.shape)) # 6

        ### add for offline to online ###
        self.last_reward = 0


    def initialize_for_new_design(self):
        self._rl_alg.episode_init()
        self._data_rewards = []
        self._episode_counter = 0


    def single_iteration(self):
        print("Time for one iteration: {}".format(time.time() - self._last_single_iteration_time))
        self._last_single_iteration_time = time.time()
        self._replay.set_mode("species")
        '''step 1: collecting the dataset'''
        current_reward = self.collect_training_experience()
        '''step 2: training population networks and individual networks'''
        train_pop = self._design_counter > 0
        if self._episode_counter >= self._config['initial_episodes']:
            self._rl_alg.single_train_step(train_ind=True, train_pop=train_pop,current_reward=current_reward,last_reward=self.last_reward)
        self._episode_counter += 1
        '''step 3: evaluate the trained network and record the data'''
        self.execute_policy()
        self.last_reward = current_reward


    def collect_training_experience(self):
        state = self._env.reset()
        nmbr_of_steps = 0
        done = False
        episode_reward_adapt = 0
        print('collecting training experience...')

        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        self._policy_cpu = policy_gpu_ind

        if self._config['use_cpu_for_rollout']:
            utils.move_to_cpu()
        else:
            utils.move_to_cuda(self._config)

        ###### SAC ######
        if self._config['rl_method'] == 'SoftActorCritic':
            while not(done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                action, _ = self._policy_cpu.get_action(state)
                new_state, reward, done, info = self._env.step(action)
                reward = reward * self._reward_scale
                terminal = np.array([done])
                reward = np.array([reward])
                self._replay.add_sample(observation=state, action=action, reward=reward, next_observation=new_state,terminal=terminal)
                state = new_state
            self._replay.terminate_episode()
            utils.move_to_cuda(self._config)

        ###### TD3 ######
        elif self._config['rl_method'] == 'TD3':
            while not(done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                deterministic_action, _ = self._policy_cpu.get_action(state)
                action = (deterministic_action) + np.random.normal(0, self.max_action * self.expl_noise,size=self.action_dim).clip(-self.max_action,self.max_action)
                new_state, reward, done, info = self._env.step(action)
                reward = reward * self._reward_scale
                terminal = np.array([done])
                reward = np.array([reward])
                episode_reward_adapt += reward
                self._replay.add_sample(observation=state, action=action, reward=reward, next_observation=new_state,terminal=terminal)
                state = new_state
            self._replay.terminate_episode()
            utils.move_to_cuda(self._config)
        else:
            raise ValueError

        return episode_reward_adapt

    def execute_policy(self):
        print('Execute policy...')
        state = self._env.reset()
        done = False
        reward_original = 0.0
        action_cost = 0.0
        nmbr_of_steps = 0
        self.reward_ep = 0.0

        if self._episode_counter < self._config['initial_episodes']:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['population'])
        else:
            policy_gpu_ind = self._rl_alg_class.get_policy_network(self._networks['individual'])
        self._policy_cpu = policy_gpu_ind

        if self._config['use_cpu_for_rollout']:
            utils.move_to_cpu()
        else:
            utils.move_to_cuda(self._config)

        ###### SAC ######
        if self._config['rl_method'] == 'SoftActorCritic':
            while not(done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                action, _ = self._policy_cpu.get_action(state, deterministic=True)
                new_state, reward, done, info = self._env.step(action)
                action_cost += info['orig_action_cost']
                self.reward_ep += float(reward)
                reward_original += float(info['orig_reward'])
                state = new_state
            utils.move_to_cuda(self._config)
            self._data_rewards.append(self.reward_ep)

        ###### TD3 ######
        elif self._config['rl_method'] == 'TD3':
            while not (done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                action, _ = self._policy_cpu.get_action(state)
                new_state, reward, done, info = self._env.step(action)
                action_cost += info['orig_action_cost']
                self.reward_ep += float(reward)
                reward_original += float(info['orig_reward'])
                state = new_state
            utils.move_to_cuda(self._config)
            self._data_rewards.append(self.reward_ep)
        else:
            raise ValueError


    def run(self):
        iterations_init = self._config['iterations_init']
        iterations = self._config['iterations']
        design_cycles = self._config['design_cycles']

        self._intial_design_loop(iterations_init)
        self._training_loop(iterations, design_cycles)

    def _training_loop(self, iterations, design_cycles):
        self.initialize_for_new_design()
        self._data_design_type = 'Optimized'

        '''get optimized parameters'''
        optimized_params = self._env.get_random_design()
        q_network = self._rl_alg_class.get_q_network(self._networks['population'])
        policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
        optimized_params = self._do_alg.optimize_design(design=optimized_params, q_network=q_network, policy_network=policy_network)
        optimized_params = list(optimized_params)

        for i in range(design_cycles):
            self._design_counter += 1
            self._env.set_new_design(optimized_params)

            '''optimizing local policy using RL'''
            for j in range(iterations):
                self.single_iteration()

            '''optimizing the design using PSO'''
            if i % 2 == 1:
                self._data_design_type = 'Optimized'
                q_network = self._rl_alg_class.get_q_network(self._networks['population'])
                policy_network = self._rl_alg_class.get_policy_network(self._networks['population'])
                optimized_params = self._do_alg.optimize_design(design=optimized_params, q_network=q_network, policy_network=policy_network)
                optimized_params = list(optimized_params)
            else:
                self._data_design_type = 'Random'
                optimized_params = self._env.get_random_design()
                optimized_params = list(optimized_params)

            self.initialize_for_new_design()

    def _intial_design_loop(self, iterations):
        self._data_design_type = 'Initial'
        for params in self._env.init_sim_params:
            self._design_counter += 1
            self._env.set_new_design(params)
            self.initialize_for_new_design()

            # Reinforcement Learning
            for i in range(iterations):
                self.single_iteration()
