import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import tempfile
import xmltodict

class HalfCheetahMujocoEnv(mujoco_env.MujocoEnv,utils.EzPickle):
    def __init__(self,config=None):
        # self._config = config
        # self._render = self._config['env']['render']
        # self._record_video = self._config['env']['record_video']

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.original_filepath = "%s/assets/half_cheetah_coadapt.xml" % dir_path

        mujoco_env.MujocoEnv.__init__(self,self.original_filepath,5)
        self.design_dim = 6

        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate(
            [(self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
              self.sim.data.qpos.flat[1:],
              self.sim.data.qvel.flat,])

    def seed(self,seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)

    def step(self,action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action,self.frame_skip)
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = ob[0]
        reward = reward_run + reward_ctrl
        done = False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq,low=-0.1,high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos,qvel)
        self.xposbefore = self.get_body_com("torso")[0]
        return self._get_obs()

    def reset_design(self,design=None):
        adapted_xml_file = tempfile.NamedTemporaryFile(delete=False,prefix='Cheetah_',suffix='.xml')
        adapted_xml_filepath = adapted_xml_file.name
        with open (self.original_filepath,'r') as fd:
            xml_string = fd.read()

            if design is None:
                bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = np.random.uniform(low=0.5,high=1.5,size=6)
            else:
                bth_r, bsh_r, bfo_r, fth_r, fsh_r, ffo_r = design

            height = max(.145 * bth_r + .15 * bsh_r + .094 * bfo_r, .133 * fth_r +  .106 * fsh_r + .07 * ffo_r)
            height *= 2.0 + 0.01

            xml_dict = xmltodict.parse(xml_string)
            xml_dict['mujoco']['worldbody']['body']['@pos'] = "0 0 {}".format(height)

            ### Btigh

            xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@pos'] = '.1 0 -.13'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@pos'] = '{} 0 {}'.format(.1 * bth_r, -.13 * bth_r)

            xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@size'] = '0.046 .145'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['geom']['@size'] = '0.046 {}'.format(.145 * bth_r)

            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['@pos'] = '.16 0 -.25'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['@pos'] = '{} 0 {}'.format(.16 * bth_r, -.25 * bth_r)

            ### bshin

            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@pos'] = '-.14 0 -.07'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@pos'] = '{} 0 {}'.format(-.14 * bsh_r,-.07 * bsh_r)

            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@size'] = '0.046 .15'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['geom']['@size'] = '0.046 {}'.format(.15 * bsh_r)

            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['@pos'] = '-.28 0 -.14'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['@pos'] = '{} 0 {}'.format(-.28 * bsh_r,-.14 * bsh_r)

            ### bfoot

            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@pos'] = '.03 0 -.097'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@pos'] = '{} 0 {}'.format(.03 * bfo_r, -.097 * bfo_r)

            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@size'] = '0.046 .094'
            xml_dict['mujoco']['worldbody']['body']['body'][0]['body']['body']['geom']['@size'] = '0.046 {}'.format(.094 * bfo_r)

            ### fthigh

            xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@pos'] = '-.07 0 -.12'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@pos'] = '{} 0 {}'.format(-.07 * fth_r,-.12 * fth_r)

            xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@size'] = '0.046 .133'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['geom']['@size'] = '0.046 {}'.format(.133 * fth_r)

            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['@pos'] = '-.14 0 -.24'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['@pos'] = '{} 0 {}'.format(-.14 * fth_r,-.24 * fth_r)

            ### fshin

            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@pos'] = '.065 0 -.09'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@pos'] = '{} 0 {}'.format(.065 * fsh_r,-.09 * fsh_r)

            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@size'] = '0.046 .106'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['geom']['@size'] = '0.046 {}'.format(.106 * fsh_r)

            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['@pos'] = '.13 0 -.18'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['@pos'] = '{} 0 {}'.format(.13 * fsh_r,-.18 * fsh_r)

            ### ffoot

            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@pos'] = '.045 0 -.07'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@pos'] = '{} 0 {}'.format(.045 * ffo_r, -.07 * ffo_r)

            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@size'] = '0.046 .07'
            xml_dict['mujoco']['worldbody']['body']['body'][1]['body']['body']['geom']['@size'] = '0.046 {}'.format(.07 * ffo_r)

        xml_string = xmltodict.unparse(xml_dict,pretty=True)
        with open(adapted_xml_filepath,'w') as fd:
            fd.write(xml_string)

        mujoco_env.MujocoEnv.__init__(self, adapted_xml_filepath, 5)
        self.reset()



if __name__ == '__main__':
    env = HalfCheetahMujocoEnv()
    design = np.array([1.5,1.0,1.2,0.8,1.5,1.0])
    env.reset_design(design)

    while True:
        # action = np.array([0] * 6)
        action = env.action_space.sample()
        ob,reward,done,info = env.step(action)
        print('The shape of ob is:',ob.shape)
        env.render()
