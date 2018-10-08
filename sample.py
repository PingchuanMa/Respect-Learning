from env import ProstheticsEnv

env = ProstheticsEnv(visualize=True, difficulty=1)
observation = env.reset()
print(env.targets[-1])
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())