import itertools
import time

from baselines import deepq
from baselines.common.models import mlp
import gym
from gym.envs.registration import register
import gym_snake


MODEL_PATH = 'data/dqn_model.pkl'
NUM_EPISODES = 100


register(
    id='Snake-v1',
    entry_point='gym_snake.envs:SnakeEnv',
    max_episode_steps=1000,
    kwargs={'width': 5, 'height': 5}
)

env = gym.make('Snake-v1')
act = deepq.learn(
    env,
    network=mlp(num_layers=2, num_hidden=100),
    total_timesteps=0,
    load_path=MODEL_PATH
)

for episode in range(NUM_EPISODES):
    observation = env.reset()
    for t in itertools.count():
        env.render()
        time.sleep(0.1)
        action = act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print('episode {} finished after {} timesteps'.format(episode, t))
            break
env.close()
