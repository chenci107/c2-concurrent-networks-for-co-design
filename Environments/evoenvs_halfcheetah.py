from gym import spaces
import numpy as np
import copy
from utils import BestEpisodesVideoRecorder
from .halfcheetah_cripple import HalfCheetahMujocoEnv

class HalfCheetahEnv(object):
    def __init__(self,config):
        self._config = config
        self._render = self._config['env']['render']
        self._record_video = self._config['env']['record_video']
        self._current_design = [1.0] * 6
        self._config_numpy = np.array(self._current_design)
        self.design_params_bounds = [(0.5,1.5),(0.5,1.5),(0.5,1.5),(0.5,1.5),(0.5,1.5),(0.5,1.5)]

        self._env = HalfCheetahMujocoEnv(config=config)

        # self._env = HalfCheetahMujocoEnvGym(config=config)

        self.init_sim_params = [[1.0,1.0,1.0,1.0,1.0,1.0],
                                [1.0,0.5,1.0,0.5,1.0,0.5],
                                [1.5,1.0,1.5,1.0,1.5,1.0],
                                [1.2,0.8,1.2,0.8,1.2,0.8],
                                [1.5,1.0,1.2,0.8,1.5,1.0]]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=[self._env.observation_space.shape[0] + self._env.design_dim],
                                            dtype=np.float32)
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()
        if self._record_video:
            self._video_recorder = BestEpisodesVideoRecorder(path=config['data_folder_experiment'],max_videos=5)
        self._design_dims = list(range(self.observation_space.shape[0] - len(self._current_design), self.observation_space.shape[0]))
        assert len(self._design_dims) == self._env.design_dim

    def render(self):
        self._env.render()

    def step(self,action):
        info = {}
        state, reward, done,_ = self._env.step(action)
        state = np.append(state,self._config_numpy)
        info['orig_action_cost'] = 0.1 * np.mean(np.square(action))
        info['orig_reward'] = reward
        if self._record_video:
            self._video_recorder.step(env=self._env,state=state,reward=reward,done=done)
        return state, reward, False, info

    def reset(self):
        state = self._env.reset()
        self._initial_state = state
        state = np.append(state,self._config_numpy)
        if self._record_video:
            self._video_recorder.reset(env=self._env,state=state,reward=0,done=False)
        return state

    def set_new_design(self,design):
        self._env.reset_design(design)
        self._current_design = design
        self._config_numpy = np.array(design)
        if self._record_video:
            self._video_recorder.increase_folder_counter()

    def get_random_design(self):
        optimized_params = np.random.uniform(low=self.design_params_bounds[0][0],high=self.design_params_bounds[0][1],size=len(self._current_design))
        return optimized_params

    def get_current_design(self):
        return copy.copy(self._current_design)

    def get_design_dimensions(self):
        return copy.copy(self._design_dims)

    def seed(self,seed):
        self._env.seed(seed)


if __name__ == '__main__':
    config = {'env':dict(render=True,record_video=False)}
    env = HalfCheetahEnv(config)
    episode = 0
    design = [[1.0,1.0,1.0,1.0,1.0,1.0],
              [1.0,0.5,1.0,0.5,1.0,0.5],
              [1.5,1.0,1.5,1.0,1.5,1.0],
              [1.2,0.8,1.2,0.8,1.2,0.8],
              [1.5,1.0,1.2,0.8,1.5,1.0]]
    for des in design:
        env.set_new_design(des)
        for i in range(100):
            action = env.action_space.sample()
            ob,reward,done,info = env.step(action)
            env.render()
