from simulator import Simulator
from models.OpAL import OpAL
from environments.bandit_task import BanditTask
import numpy as np

ps = np.array([0.3, 0.7])
start_ = np.zeros((1, ), dtype=np.int32)
steps_ = 1000
lr_ = 0.01
temperature_ = 1.0

environment_ = BanditTask(ps, start=start_)
model_ = OpAL(actions=ps, domain_shape=[1], state=start_, lr_n=lr_, lr_g=lr_, temperature=temperature_)
interaction_space_ = np.array(range(len(ps)), dtype=np.int32)

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_)
res = sim.simulate(steps=steps_)
print(res)
