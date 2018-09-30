from env import ProstheticsEnv
from util import get_mirror_id

env = ProstheticsEnv(visualize=False)
env.reset()
mirror_id = get_mirror_id(env.get_state_desc())

print(mirror_id)
print(len(mirror_id[0]))
print(len(mirror_id[1]))
print(len(mirror_id[2]))