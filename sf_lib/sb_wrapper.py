import gym
from gym.spaces import Discrete, MultiBinary, Box
import numpy as np
import cv2

NUM_ACTIONS = 17

def sf_discretize(action):
    mb_array = np.zeros(12)
    if action == 0:
        pass
    elif action == 1:
        mb_array[10] = 1
    elif action == 2:
        mb_array[9] = 1
    elif action == 3:
        mb_array[11] = 1
    elif action == 4:
        mb_array[1] = 1
    elif action == 5:
        mb_array[0] = 1
    elif action == 6:
        mb_array[8] = 1
    elif action == 7:
        mb_array[4] = 1
    elif action == 8:
        mb_array[4] = 1
        mb_array[7] = 1
    elif action == 9:
        mb_array[7] = 1
    elif action == 10:
        mb_array[7] = 1
        mb_array[5] = 1
    elif action == 11:
        mb_array[5] = 1
    elif action == 12:
        mb_array[5] = 1
        mb_array[6] = 1
    elif action == 13:
        mb_array[6] = 1
    elif action == 14:
        mb_array[6] = 1
        mb_array[4] = 1
    elif action == 15:
        mb_array[7] = 1
        mb_array[11] = 1
    elif action == 16:
        mb_array[6] = 1
        mb_array[11] = 1
    else:
        print("Invalid input for discrete action")
    return mb_array

class SF_Random_Discretizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(NUM_ACTIONS)

    def step(self,action):
        p1_action = sf_discretize(action)

        obs, reward, done, info = self.env.step(p1_action.tolist() + MultiBinary(12).sample().tolist())

        return obs, reward[0], done, info


class SF_Stationary_Discretizer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(NUM_ACTIONS)

    def step(self,action):
        p1_action = sf_discretize(action)

        obs, reward, done, info = self.env.step(p1_action.tolist() + np.zeros(12).tolist())

        return obs, reward[0], done, info


class SFWarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=100, height=100, grayscale=False):
        """
        Warps frames to specified height and with, has option for grayscale, and
        crops the top so no health bars or timer is showing.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.grayscale = grayscale
        if self.grayscale:
            self.num_colors = 1
        else:
            self.num_colors = 3
        self.width = width
        self.height = height
        self.crop = int(height/5)
        self.observation_space = Box(low=0, high=255, shape=(self.height-self.crop, self.width, self.num_colors),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        if self.grayscale:
             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        crop_frame = frame[self.crop:, :]
        if self.grayscale:
            crop_frame = crop_frame[:,:, None]
        return crop_frame


# class SFRandomWrapper(gym.Wrapper):
#
#     def __init__(self,env):
#         super().__init__(env)
#
#     def step(self, action):
#         action_plus_random = action.tolist() + MultiBinary(12).sample().tolist()
#         return self.env.step(action_plus_random)