import gym
import highway_env
from RLAgents.PPO.PPO import PPOAgent
from RLAgents.DQN.DQNAgent import DQNAgent
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train", action="store_true", default=False, help="Train agent")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from existing model")
parser.add_argument("--test", action="store_true", default=False, help="Test existing model")
settings = parser.parse_args()

env_name = 'intersection-v0'

obs = "Kinematics"
# obs = "Image"

# action = "ContinuousAction"
action = "DiscreteMetaAction"

# algorithm = "PPO"
algorithm = "DQN"


def env():
    env = gym.make(env_name)
    if obs == "Kinematics":
        env.configure({
            "offroad_terminal": True,
            "simulation_frequency": 8,
            "duration": 13,
            "policy_frequency": 4,
            "offscreen_rendering": True
        })
    elif obs == "Image":
        env.configure({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 128),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.5
            },
            "offroad_terminal": True,
            "simulation_frequency": 8,
            "duration": 240,
            "policy_frequency": 4,
            "offscreen_rendering": True
        })

    if action == "DiscreteMetaAction":
        env.configure({"action": {
             "type": "DiscreteMetaAction"
        }})
    else:
        env.configure({"action": {
             "type": "ContinuousAction"
        }})
    env.reset()
    return env


if settings.train:
    if algorithm == "DQN":
        agent = DQNAgent(env,
                         model_key="dqn_{}_{}_{}".format(env_name, obs, action),
                         obs=obs,
                         use_double_dqn=False, use_dueling_dqn=False, use_priority=False,
                         normalize_reward_coeff=1)
        agent.learn()
    else:
        agent = PPOAgent(env, obs=obs, resume=settings.resume,
                         model_key="ppo_{}_{}_{}".format(env_name, obs, action))
        agent.play(test=False, save_model=True)

elif settings.test:
    if algorithm == "DQN":
        agent = DQNAgent(env,
                         model_key="dqn_{}_{}_{}".format(env_name, obs, action),
                         obs=obs,
                         resume=True,
                         use_double_dqn=False, use_dueling_dqn=False, use_priority=False,
                         normalize_reward_coeff=1)
        agent.test_policy(games=1, render=True, save=True)
    else:
        agent = PPOAgent(env, obs=obs, resume=True,
                         model_key="ppo_{}_{}_{}".format(env_name, obs, action))

    for _ in range(1):
        score, _ = agent.test_play(games=1, gui=True, save=True)
        print(score, len(_[0]))
