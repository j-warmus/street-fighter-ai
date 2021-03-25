from stable_baselines import A2C
import time
def evaluate(modelname,env):
    n_cores = 4
    obs = env.reset()
    model = A2C.load(modelname)
    wr = 0
    win = 0
    total_health_diff = 0
    loss = 0
    episodes = 0
    total_episodes = 100
    while episodes < total_episodes:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #print(rewards[0])
        time.sleep(.04)
        # print(rewards[0])
        env.render(mode="human")
        for i in range(4):

            if dones[i] == True:
                if info[i]["p1_health"] < info[i]["p2_health"]:
                    loss += 1
                else:
                    win += 1
                total_health_diff += info[i]["p1_health"] - info[i]["p2_health"]
                wr = win/(win+loss)
                episodes += 1
    return wr, total_health_diff/total_episodes

