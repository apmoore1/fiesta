'''
Module that contains useful functions that are used within the main fiesta
module.

Functions:

1. pull_arm -- 
'''

import numpy as np

def pull_arm(mean: float, sd: float) -> float:
    '''
    It will return a value that has been sampled from a normal distribution 
    that has a mean and standard deviation of those given as arguments.

    :param mean: The mean of the normal distribution.
    :param sd: The standard deviation of the normal distribution.
    :returns: A value that has been sampled from a normal distribution that has
    a mean and standard deviation of those given as arguments.
    '''

    return np.random.normal(mean, sd)