from baselines import deepq
from baselines.common.models import mlp
import gym
from gym.envs.registration import register
import gym_snake


MODEL_PATH = '../data/dqn_model.pkl'


def main():
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
        total_timesteps=1000000,
        exploration_fraction=0.1
    )
    print(f'Saving model to {MODEL_PATH}')
    act.save(MODEL_PATH)


if __name__ == '__main__':
    main()
