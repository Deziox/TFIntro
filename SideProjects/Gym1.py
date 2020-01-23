import gym
from gym import envs

print(envs.registry.all())

env = gym.make('CartPole-v0')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())
    print(env.action_space.sample())