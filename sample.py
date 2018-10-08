from env import ProstheticsEnv
from util import state_desc_to_ob

env = ProstheticsEnv(visualize=True, difficulty=1, no_acc=True)
observation = env.reset()
print(len(observation))
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())