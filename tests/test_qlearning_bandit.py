from simulator import Simulator
from models.q_learning import QLearning
from environments.bandit_task import BanditTask

ps = [0.3, 0.7]
n_options_ = len(ps)
steps_ = 1000
lr_ = 0.01
temperature_ = 1.0

environment_ = BanditTask(ps)
model_ = QLearning(n_options=n_options_, lr=lr_, temperature=temperature_, steps=steps_)

sim = Simulator(model=model_, environment=environment_)
res = sim.simulate(steps=steps_)
print(res)
