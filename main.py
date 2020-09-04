import gym
from gym.wrappers import AtariPreprocessing
from DuelDDQN_Agent import Agent
import numpy as np

env = gym.make("BreakoutNoFrameskip-v4")
env = AtariPreprocessing(env, grayscale_obs=False)
s = env.reset()
num_episodes = 10
agent = Agent(3, 4)
scores = []
for i in range(num_episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(np.moveaxis(state, -1, 0))
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(np.moveaxis(state, -1, 0), action, reward, np.moveaxis(next_state, -1, 0), done)
        agent.learn()
        score += reward
    scores.append(score)
    if max(scores) <= score:
        agent.save_models()
    print(f"Episode {i}, Score {score}, Epsilon {agent.epsilon}")
