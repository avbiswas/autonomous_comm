import gym
import highway_env
from RLAgents.PPO import PPOAgent
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true", default=False, help="Train agent")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from existing model")
parser.add_argument("--test", action="store_true", default=False, help="Test existing model")
settings = parser.parse_args()

env = gym.make("highway-v0")

env.configure({
    "action": {
        "type": "ContinuousAction"
    },
    "offroad_terminal": True,
    "simulation_frequency": 4,
    "policy_frequency": 1
})
env.reset()

'''
for _ in range(10):
    s, r, d, _ = env.step(env.action_space.sample())
    print(s)
'''
# obs = "OccupancyGrid"
# obs = "Kinematics"
obs = "Image"

if settings.train:
    render = True
    agent = PPOAgent(env, obs=obs, resume=settings.resume)
    agent.play(test=False, save_model=True)

elif settings.test:
    agent = PPOAgent(env, obs=obs, resume=True)
    for _ in range(1):
        score, _ = agent.test_play(games=1, gui=True, save=True)
        print(score)
