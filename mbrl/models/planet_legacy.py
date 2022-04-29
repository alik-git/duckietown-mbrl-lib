###################
##### FOR GLEN ####
###################
# This implementation of PlaNet servers as the world model for our Dreamer
# implementation.

###################
##### FOR GLEN ####
###################
# This was used to do QA on the PlaNet implementation we ported from 
# Chandramouli Rajagopalan’s pytorch implementation of Dreamer here 
# https://github.com/chamorajg/pl-dreamer , which has helper functions for the 
# non-MBRL-Lib PlaNet, but do not get used with MBRL-Lib or Duckietown. 
######################################
import torch
import numpy as np
from typing import Tuple, Any, Union, Optional, Dict
import gym
TensorType = Any

class DMControlSuiteEnv:

    def __init__(self, 
                name: str, 
                max_episode_length: int = 1000,
                action_repeat:int = 2,
                size: Tuple[int] = (64, 64),
                camera: Optional[Any] = None,
                ):
        domain, task = name.split('_', 1)
        if domain == 'cup':
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self._step = 0
        self._max_episode_length = max_episode_length
        self._action_repeat = action_repeat
    
    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
            spaces['image'] = gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        reward = 0
        obs = None
        for k in range(self._action_repeat):
            time_step = self._env.step(action)
            self._step += 1
            obs = dict(time_step.observation)
            obs['image'] = self.render()
            reward += time_step.reward or 0
            done = time_step.last() or self._step == self._max_episode_length
            if done:
                break
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs["image"], reward, done, info

    def reset(self):
        time_step = self._env.reset()
        self._step = 0
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs["image"]

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)

class Episode(object):
    """ Episode Class which contains the related
        attributes of an environment episode in the
        the format similar to queue"""
    
    def __init__(self,
                obs:TensorType,
                action_space: int = 1,
                action_repeat: int = 2) -> None:
        """Initializes a list of all episode attributes"""
        self.action_space = action_space
        self.action_repeat = action_repeat
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose
                                                        ((2, 0, 1))))
        self.t = 1
        self.obs = [obs]
        self.action = [torch.FloatTensor(torch.zeros(1, self.action_space)).squeeze()]
        self.reward = [0]
        self.done = [False]
    
    def append(self, 
                episode_attrs: Tuple[TensorType]) -> None:
        """ Appends episode attribute to the list."""
        obs, action, reward, done = episode_attrs
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose
                                                        ((2, 0, 1))))
        self.t += 1
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
    
    def reset(self, 
            obs:TensorType) -> None:
        """ Resets Episode list of attributes."""
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose
                                                        ((2, 0, 1))))
        self.t = 1
        self.obs = [obs]
        self.action = [torch.FloatTensor(torch.zeros(1, self.action_space)).squeeze()]
        self.reward = [0]
        self.done = [False]
    
    def todict(self,) -> Dict:
        episode_dict = dict({'count': self.t,
                                'obs': torch.stack(self.obs),
                                'action': torch.cat(self.action),
                                'reward': torch.FloatTensor(self.reward),
                                'done': torch.BoolTensor(self.done)})
        return 