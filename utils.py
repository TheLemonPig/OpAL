import numpy as np


def tempered_softmax(choices, temperature):
    p_values = np.exp(choices/temperature)/np.exp(choices/temperature).sum()
    return p_values

