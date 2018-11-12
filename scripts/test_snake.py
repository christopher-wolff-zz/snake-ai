import gym
import gym_snake

env = gym.make('Snake-v0')

for i in range(100):
    env.reset()
    for t in range(1000):
        env.render()
        observation, reward, done, _ = env.step(env.action_space.sample())
        print(observation.shape)
        if done:
            print('episode {} finished after {} timesteps'.format(i, t))
            break
