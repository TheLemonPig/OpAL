from simulator import Simulator
from models.actor_critic import ActorCritic
from environments.grid_world import GridWorld
import numpy as np

actions = np.array([0.8, 0.2, 0, 0.2])
world_array_ = np.ones((3, 4)) * -0.04
terminals_ = {(2, 3): 0.7, (0, 3): 0.8}
success_state = (0, 3)
failure_state = (2, 3)
for k, v in terminals_.items():
    world_array_[k] = v
start_ = (1, 0)
steps_ = 10000
alpha_ = 0.1
beta_ = 1.0
gamma_ = 0.9
temperature_ = 1

environment_ = GridWorld(world_array=world_array_)
model_ = ActorCritic(actions=actions, domain_shape=world_array_.shape, state=start_, alpha=alpha_, beta=beta_,
                     gamma=gamma_, temperature=temperature_)
interaction_space_ = [(1, 0), (0, 1), (-1, 0), (0, -1)]

sim = Simulator(model=model_, environment=environment_, interaction_space=interaction_space_, terminals=terminals_)
res = sim.simulate(steps=steps_)
pred = sim.get_predictions()
for i in range(len(actions)):
    print(pred["ps"][..., i])
    print(pred["vs"][..., i])
rewards = res['rewards']
states = res['states']
rewards = rewards[-len(rewards)//10:]
success = states.count(success_state)
failure = states.count(failure_state)
total = success + failure
print(f'\nSuccess rate: {np.round(success/total*100,2)}%\nFailure rate: {np.round(failure/total*100,2)}%\n')
print(f'Average: {np.round(sum(rewards)/len(rewards),2)}')
policy = model_.get_optimal_policy()
for t in terminals_.keys():
    policy[t] = -1
print(policy)
print(world_array_)

