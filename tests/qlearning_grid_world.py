from simulator import Simulator
from models.q_learning import QLearning
from environments.grid_world import GridWorld
import numpy as np

actions = np.array([0.8, 0.2, 0, 0.2])
world_array_ = np.ones((3, 4)) * -0.04
world_array_[2, 3] = 1
world_array_[2, 0] = -1
steps_ = 10000
lr_ = 0.01
temperature_ = 1.0

environment_ = GridWorld(world_array=world_array_)
start_ = environment_.get_start()
model_ = QLearning(actions=actions, domain_shape=world_array_.shape, state=start_, lr=lr_, temperature=temperature_)
interaction_space_ = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_)
res = sim.simulate(steps=steps_)
for i in range(len(actions)+1):
    print(res[..., i])
# for n in range(steps_):
#     action = model_.act()
#     new_state, reward = environment_.interact(state, action)
#     model_.update(action, new_state, reward)

if __name__ == "__main__":
    ...