from simulator import Simulator
from models.q_learning import QLearning
from environments.bandit_task import BanditTask
import numpy as np

ps = np.array([0.3, 0.7])
start_ = 0
steps_ = 1000
lr_ = 0.01
temperature_ = 1.0

environment_ = BanditTask(ps, start=start_)
model_ = QLearning(actions=ps, domain_shape=[1], state=start_, lr=lr_, temperature=temperature_)
interaction_space_ = np.array(range(len(ps)), dtype=np.int32)

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_)
res = sim.simulate(steps=steps_)
print(res)
