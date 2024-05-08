import numpy as np


def safe_softmax(choices):
    p_values = np.exp(choices-choices.max())/np.exp(choices-choices.max()).sum()
    return p_values


def tempered_softmax(choices, temperature):
    # cap choice values to avoid overflow
    # choices[choices > 10] = 10
    # calculate softmax over choices, weighted by temperature
    p_values = np.exp(choices/temperature)/np.exp(choices/temperature).sum()
    return p_values


def location_counter(state_list, domain):
    location_counts = np.zeros(domain)
    x, y = domain
    for i in range(x):
        for j in range(y):
            location_counts[i, j] += state_list.count((i, j))
    return location_counts


def state_action_counter(state_action_list, state_action_space):
    action_counts = np.zeros(state_action_space)
    for i in range(state_action_space[0]):
        for j in range(state_action_space[1]):
            action_counts[i, j] += state_action_list.count((i, j))
    return action_counts


def state_action_counter(state_action_list, state_action_space):
    state_action_counts = np.zeros(state_action_space)
    (x, y, a) = state_action_space
    for i in range(x):
        for j in range(y):
            for k in range(a):
                state_action_counts[i, j, k] += state_action_list.count((i, j, k))
    return state_action_counts


def cumulative_reward(rewards):
    cumulative = []
    for n in range(len(rewards)-1):
        cumulative.append(sum(rewards[:n]))
    cumulative.append(sum(rewards))
    return cumulative


def get_square_triangles(x, y, size):
    half_size = size / 2
    triangle1 = [(x + size, y), (x + half_size, y + half_size), (x, y), (x + size, y)]
    triangle2 = [(x + size, y + size), (x + half_size, y + half_size), (x + size, y), (x + size, y + size)]
    triangle3 = [(x, y + size), (x + half_size, y + half_size), (x + size, y + size), (x, y + size)]
    triangle4 = [(x, y), (x + half_size, y + half_size), (x, y + size), (x, y)]
    return [triangle1, triangle2, triangle3, triangle4]