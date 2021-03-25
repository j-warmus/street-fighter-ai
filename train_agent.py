import retro
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,MlpLnLstmPolicy,CnnPolicy,CnnLstmPolicy,CnnLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv,DummyVecEnv,VecFrameStack
from stable_baselines import PPO2, A2C, DQN
from stable_baselines.common.atari_wrappers import MaxAndSkipEnv
from sf_lib.sb_wrapper import SF_Random_Discretizer, SFWarpFrame
from sf_lib.wrapperfacing import SFRamObsDiscActWrapper
from gym.wrappers.frame_stack import FrameStack
from sf_lib.custom_policies import CustomLSTMPolicy
from sf_lib.eval_wr import evaluate

if __name__ == '__main__':
    mode = 0
    gamename = 'StreetFighterIISpecialChampionEdition-Genesis'
    n_cores = 4
    modelname = 'SFII-A2C-c2'
    # apply wrappers here to customize the learning environment.  This uses the discretizer and the frame skipper,
    # with ram observations.  SubprocVecEnv is required by stable baselines.
    env = SubprocVecEnv(
        [lambda: MaxAndSkipEnv(SFRamObsDiscActWrapper(retro.make(gamename, players=2), mode = "random"),skip = 6) for i in range(n_cores)]
    )

    print(env.observation_space)

    if mode == 0:
        model = A2C(CustomLSTMPolicy,env,verbose=1,n_steps=6,gamma=0.9, tensorboard_log="vchun")
        model.set_env(env)
        model.learn(total_timesteps=5000)
        model.save(modelname)
        print(evaluate(modelname, env))
    elif mode == 1:
        print(evaluate(modelname,env))

