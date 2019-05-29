'''
Module that contains useful functions that are used within the main fiesta
module.

Functions:

1. pull_arm -- return a value that has been sampled from a normal distribution 
   that has a mean and standard deviation of those given as arguments.
2. belief_calc -- The number of times a model performed best for each sample 
   based on each model belief distribution, normliased by the number of samples 
   (this is in affect the confidence for each model of whether that model is 
   the best model p-value can be calculated by 1 - the confidence value). 
   shape (Number of models,)
3. lists_same_size -- Checks if the lists given as arguments are of the same 
   size, if not it raises a ValueError.
'''
from typing import List

import numpy as np

def pull_arm(mean: float, sd: float) -> float:
    '''
    :param mean: The mean of the normal distribution.
    :param sd: The standard deviation of the normal distribution.
    :returns: A value that has been sampled from a normal distribution that has
    a mean and standard deviation of those given as arguments.
    '''

    return np.random.normal(mean, sd)

def belief_calc(est_means: np.ndarray, est_variances: np.ndarray, 
                eval_counts: np.ndarray, samples: int = 100000) -> List[float]:
    '''
    :param est_means: A vector of each models mean scores 
                      shape: (Number of models,) 
    :param est_means: A vector of each models variance of there scores:
                      shape: (Number of models,) 
    :param eval_counts: A vector stating the number of times each model has 
                        been evaluated/ran: shape: (Number of models,)
    :param samples: Number of samples to generate from our belief distribution 
                    for each model. This should be large e.g. minimum 10000
    :returns: The number of times a model performed best for each sample 
              based on each model belief distribution, normliased by the 
              number of samples (this is in affect the confidence for each 
              model of whether that model is the best model p-value can be 
              calculated by 1 - the confidence value). 
              shape (Number of models,)
    :raises ValueError: If the eval_counts contains values less than 3. As 
                        it is required that each model has been evaluated a
                        minimum of 3 times, this is due to our prior beleif 
                        in our algorthim.  
    '''

    est_sd = np.sqrt(est_variances)
    num_models = eval_counts.shape[0]
    if sum((eval_counts - 2.0) > 0) != num_models:
        raise ValueError("The number of times each model has been evaluated "
                         "has to be greater than 2, at least one of your "
                         f"model's has been evaluated less:{eval_counts}")

    # generate samples from t dist with required degrees of freedom
    # we subtract 2 from the number of times each model has been evaluated due 
    # to our prior belief about the uniform distribution.
    t_samples = np.random.standard_t(eval_counts - 2.0, (samples, num_models))
    # scale to get samples from posterior about means
    posterior_samples= t_samples * est_sd * np.sqrt(1 / (eval_counts - 2.0)) + est_means
    # find argmax for each set of samples
    args = np.argmax(posterior_samples, axis=1)
    # count raw frequencies of maximum over samples
    max_model_index, num_times_max = np.unique(args, return_counts=True)

    pi = [0] * num_models
    for model_index in range(0, num_models):
        if model_index in max_model_index:
            num_times_index = np.where(max_model_index==model_index)[0][0]
            pi[model_index] = num_times_max[num_times_index] / samples
        else:
            pi[model_index] = 0 / samples
    return pi

def lists_same_size(*lists) -> None:
    '''
    Checks if the lists given as arguments are of the same size, if not it 
    raises a ValueError.

    :raises ValueError: If any of the lists are not the same size
    '''
    list_length = -1
    for _list in lists:
        current_list_length = len(_list)
        if list_length == -1:
            list_length = current_list_length
        else:
            if list_length != current_list_length:
                raise ValueError(f'The lists are not of the same size: {lists}')