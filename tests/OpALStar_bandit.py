from simulator import Simulator
from models.OpALStar import OpALStar
from environments.bandit_task import BanditTask
import numpy as np

ps = np.array([0.3, 0.9])
start_ = 0
steps_ = 1000
lr_ = 0.01
beta_ = 1.0
gamma_ = 0.
rho_ = 0.
phi_ = 1.0
k_ = 1.0
T_ = 100
r_mag_ = 1
l_mag_ = -1

environment_ = BanditTask(ps, start=start_)
model_ = OpALStar(actions=ps, domain_shape=[1], state=start_, alpha_c=lr_, alpha_g=lr_, alpha_n=lr_,
                  beta=beta_, gamma=gamma_, rho=rho_, phi=phi_, k=k_, T=T_, r_mag=r_mag_, l_mag=l_mag_)
interaction_space_ = np.array(range(len(ps)), dtype=np.int32)

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_)
res = sim.simulate(steps=steps_)
print(res)
print(f"Success rate: {100*np.mean(res[1])}%")
print(model_.get_optimal_policy())
