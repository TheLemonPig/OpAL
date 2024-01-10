from simulator import Simulator
from models.OpAL import OpAL
from environments.grid_world import GridWorld
import numpy as np

actions = np.array([0.8, 0.2, 0, 0.2])
world_array_ = np.ones((3, 4)) * -0.04
terminals_ = {(2, 3): 1, (2, 0): -1}
for k, v in terminals_.items():
    world_array_[k] = v
steps_ = 1000
lr_ = 0.01
temperature_ = 1.0

environment_ = GridWorld(world_array=world_array_)
start_ = environment_.get_start()
model_ = \
    OpAL(actions=actions, domain_shape=world_array_.shape, state=start_, lr_n=lr_, lr_g=lr_, temperature=temperature_)
interaction_space_ = [(1, 0), (0, 1), (-1, 0), (0, -1)]

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_, terminals=terminals_)
res = sim.simulate(steps=steps_)
for i in range(len(actions)):
    print(res[0]["gs"][..., i])
    print(res[0]["ns"][..., i])
print(res[1])
