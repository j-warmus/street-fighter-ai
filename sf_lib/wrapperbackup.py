import gym
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple

NUM_OBS = 18

def preprocess_info(info_dict, player=1):
    # info_dict is the info returned from env.step(), player is the player being played as
    # observations are returned as one hot or scaled
    # processed_info = (distance, hitstun)
    if player == 1:
        sel = "p1"
        opp = "p2"
    elif player == 2:
        sel = "p2"
        opp = "p1"
    else:
        raise ValueError("Invalid player # given")

    self_x = info_dict[sel + "_X"]
    opp_x = info_dict[sel + "_X"]
    distance = self_x - opp_x
    distance = distance / 210

    health_diff = (info_dict[sel + "_health"] - info_dict[opp + "_health"])/176

    self_y = info_dict[sel + "_Y"]
    self_y = abs((self_y - 192) / (-67))

    opp_y = info_dict[opp + "_Y"]
    opp_y = abs((opp_y - 192) / (-70))

    self_hitstun = self_recovery = opp_hitstun = opp_recovery = opp_crouch = opp_jump = opp_normal = opp_air_normal = opp_special = 0
    self_fireball_out = self_fireball_loc = opp_fireball_out = opp_fireball_loc = 0

    self_ram_state = info_dict[sel + "_state"]
    self_ram_recovery = info_dict[sel + "_recovery"]
    opp_ram_state = info_dict[opp + "_state"]
    opp_ram_recovery = info_dict[opp + "_recovery"]
    self_ram_fireball_loc = info_dict[sel + "_fireball_loc"]
    self_fireball_out = info_dict[sel + "_fireball_out"]
    opp_ram_fireball_loc = info_dict[opp + "_fireball_loc"]
    opp_fireball_out = info_dict[opp + "_fireball_out"]

    if self_ram_state == 14:
        self_hitstun = 1

    if self_ram_recovery == 1 and self_ram_state != 4:
        self_recovery = 1

    if opp_ram_state == 14:
        opp_hitstun = 1

    if opp_ram_recovery == 1 and opp_ram_state != 4:
        opp_recovery = 1

    if opp_ram_state == 2:
        opp_crouch = 1

    if opp_ram_state == 4:
        opp_jump = 1

    if opp_ram_state == 10:
        opp_normal = 1

    if opp_ram_recovery == 0 and opp_ram_state == 4:
        opp_air_normal = 1

    if opp_ram_state == 12:
        opp_special = 1

    if self_fireball_out == 1:
        self_fireball_loc = self_ram_fireball_loc
        self_fireball_loc = (self_fireball_loc - self_x - 55)/403

    if opp_fireball_out == 1:
        opp_fireball_loc = opp_ram_fireball_loc
        opp_fireball_loc = (opp_fireball_loc - self_x - 55) / 403

    self_x = (self_x - 55)/403
    opp_x = (opp_x - 55) / 403


    result = np.zeros(NUM_OBS)
    idx = 0
    for item in (distance, self_x, opp_x, self_y, opp_y, self_hitstun, self_recovery, opp_hitstun, opp_recovery,
                 opp_crouch, opp_jump, opp_normal, opp_air_normal, opp_special, self_fireball_out,
                 self_fireball_loc, opp_fireball_out, opp_fireball_loc):
        result[idx] = item
        idx += 1
    return result


def disc_to_mb(disc_action):
    # actions: none, l_P, m_P, h_P, l_K, m_K, h_K, u, ur, r, dr, d, dl, l, ul, r-HP, l-HP
    # multibinary buttons: ["B", "A", 'MODE', 'START', "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    mb_array = np.zeros(12)
    if disc_action == 0:
        pass
    elif disc_action == 1:
        mb_array[10] = 1
    elif disc_action == 2:
        mb_array[9] = 1
    elif disc_action == 3:
        mb_array[11] = 1
    elif disc_action == 4:
        mb_array[1] = 1
    elif disc_action == 5:
        mb_array[0] = 1
    elif disc_action == 6:
        mb_array[8] = 1
    elif disc_action == 7:
        mb_array[4] = 1
    elif disc_action == 8:
        mb_array[4] = 1
        mb_array[7] = 1
    elif disc_action == 9:
        mb_array[7] = 1
    elif disc_action == 10:
        mb_array[7] = 1
        mb_array[5] = 1
    elif disc_action == 11:
        mb_array[5] = 1
    elif disc_action == 12:
        mb_array[5] = 1
        mb_array[6] = 1
    elif disc_action == 13:
        mb_array[6] = 1
    elif disc_action == 14:
        mb_array[6] = 1
        mb_array[4] = 1
    elif disc_action == 15:
        mb_array[7] = 1
        mb_array[11] = 1
    elif disc_action == 16:
        mb_array[6] = 1
        mb_array[11] = 1
    else:
        print("Invalid input for discrete action")
    return mb_array


class SFIIWrapper_Disc(gym.Wrapper):
    # wrapper to allow custom observations to be returned.
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = Discrete(17)
        self.observation_space = Box(-1, 1, shape=(NUM_OBS,))
    # right now random, will need to generalize to train two agents against each other.

    def step(self, p1_action):
        p1_action = disc_to_mb(p1_action)

        obs, reward, done, info = self.env.step(p1_action.tolist() + MultiBinary(12).sample().tolist())

        info_clean = preprocess_info(info, 1)
        return info_clean,reward[0],done,info

    def reset(self, **kwargs):
        obs = self.env.reset()
        result = np.zeros(NUM_OBS)
        return result


class VSRandomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
