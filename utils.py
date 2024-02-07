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

