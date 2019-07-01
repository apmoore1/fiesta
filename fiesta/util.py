from typing import List, Callable, Any, Dict, Tuple

import numpy as np

import fiesta

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
              model of whether that model is the best model, p-value can be 
              calculated by 1 - the confidence value for each model). 
              shape (Number of models,)
    :raises ValueError: If the ``eval_counts`` contains values less than 3. As 
                        it is required that each model has been evaluated a
                        minimum of 3 times, this is due to our prior beleif 
                        in our algorthim.  
    '''

    est_sd = np.sqrt(est_variances)
    num_models = eval_counts.shape[0]
    if sum((eval_counts - 2.0) > 0) != num_models:
        raise ValueError("The number of times each model has been evaluated "
                         "has to be greater than 2 or at least one of your "
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

def fc_func_stats(N: int, correct_model_index: int, 
                  fc_func_name: str, **fc_kwargs) -> Tuple[int, int, int, float]:
    '''
    :param N: Number of times to run the Fixed Confidence (FC) function
    :param correct_model_index: The index of the best model
    :param model_funcs: Functions that will generated model evaluation scores
    :param fc_func_name: The name of the FC function being evaluated e.g. 
                        `non_adaptive_fc` or `TTTS`
    :param fc_kwargs: Keyword arguments to give to the FC function.
    :returns: Tuple containing: 
              
              1. min
              2. mean
              3. max
              4. percentage correct 
              
              Where the first 3 relate to the number of evaluations the FC 
              function required to get to the confidence level required. 
              The last is the percentage of those runs where the most 
              confident model was the correct model determined by the 
              ``correct_model_index`` argument.
    :raises ValueError: If the ``fc_func_name`` is not in the list of acceptable 
                        FC function names.
    '''
    # Get FC function
    fc_func_names = ['TTTS', 'non_adaptive_fc']
    if fc_func_name not in fc_func_names:
        raise ValueError(f'The FC function name given {fc_func_name} is not '
                        f'the list of acceptable function names {fc_func_names}')
    fc_func = getattr(fiesta, fc_func_name)
    
    # Run the FC function N times and count the number of times 
    # the best model was correctly chosen
    number_correct = 0
    total_num_evals = []
    for _ in range(0, N):
        conf_scores, _, num_evals, _ = fc_func(**fc_kwargs)
        total_num_evals.append(num_evals)
        if np.argmax(conf_scores) == correct_model_index:
            number_correct += 1
    # Summary stats
    _min = np.min(total_num_evals)
    _mean = np.mean(total_num_evals)
    _max = np.max(total_num_evals)
    perecent_correct = (number_correct / N) * 100
    return _min, _mean, _max, perecent_correct

def fb_func_stats(N: int, correct_model_index: int, 
                  fb_func_name: str, **fb_kwargs) -> float:
    '''
    :param N: Number of times to run the Fixed Budget (FB) function.
    :param correct_model_index: The index of the best model.
    :param fb_func_name: The name of the FB function being evaluated e.g. 
                         `sequential_halving` or `non_adaptive_fb`
    :param fb_kwargs: Keyword arguments to give to the FB function.
    :returns: The probability the best model was correctly identified by the 
              FB function across the N runs.
    :raises ValueError: If the ``fb_func_name`` is not in the list of acceptable 
                        FB function names.
    '''
    # Get FB function
    fb_func_names = ['sequential_halving', 'non_adaptive_fb']
    if fb_func_name not in fb_func_names:
        raise ValueError(f'The FB function name given {fb_func_name} is not '
                        f'the list of acceptable function names {fb_func_names}')
    fb_func = getattr(fiesta, fb_func_name)

    # Run the FB function N times and count the number of times 
    # the best model was correctly chosen
    number_correct = 0
    for _ in range(0, N):
        fb_return = fb_func(**fb_kwargs)
        pred_best_model_index = fb_return[0]
        if pred_best_model_index == correct_model_index:
            number_correct += 1
    probability_correct = number_correct / N
    return probability_correct