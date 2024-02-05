from simulator import Simulator
from models.actor_critic import ActorCritic
from environments.bandit_task import BanditTask
import numpy as np

ps = np.array([0.3, 0.7])
start_ = 0
steps_ = 1000
alpha_ = 0.1
beta_ = 1.0
gamma_ = 0.0
temperature_ = 1.0

environment_ = BanditTask(ps, start=start_)
model_ = ActorCritic(actions=ps, domain_shape=[1], state=start_, alpha=alpha_, beta=beta_, gamma=gamma_,
                     temperature=temperature_)
interaction_space_ = np.array(range(len(ps)), dtype=np.int32)

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_)
res = sim.simulate(steps=steps_)
print(res)
print(model_.get_optimal_policy())
