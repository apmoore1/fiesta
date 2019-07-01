import logging
from typing import List, Dict, Any, Tuple, Callable
import math

import numpy as np
from scipy.special import logit

from fiesta.util import belief_calc

logger = logging.getLogger(__name__)

def TTTS(data: List[Dict[str, Any]], 
         model_functions: List[Callable[[List[Dict[str, Any]],
                                         List[Dict[str, Any]]], float]],
         split_function: Callable[[List[Dict[str, Any]]], 
                                  Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
         p_value: float, logit_transform: bool = False, samples: int = 100000
         ) -> Tuple[List[float], List[float], int, List[List[float]]]:
    '''
    :param data: A list of dictionaries, that as a whole represents the entire 
                 dataset. Each dictionary within the list represents 
                 one sample from the dataset.
    :param model_functions: A list of functions that represent different 
                            models e.g. pytorch model. Which take a train and  
                            test dataset as input and returns a metric score 
                            e.g. Accuracy. The model functions should not have 
                            random seeds set else it defeats the point of 
                            finding the best model independent of the random 
                            seed and data split.
    :param split_function: A function that can be used to split the data into 
                           train and test splits. This should produce random 
                           splits each time it is called. If you would like to 
                           use a fixed split each time, you can hard code this 
                           function to produce a fixed split each time.
    :param p_value: The significance value for the best model to be truely the 
                    best model e.g. 0.05 if you want to be at least 
                    95% confident.
    :param logit_transform: Whether to transform the model function's returned 
                            metric score by the logit function.
    :param samples: Number of samples to generate from our belief distribution 
                    for each model. This argument is passed directly to 
                    :func:`fiesta.util.belief_calc` within this function. 
                    This should be large e.g. minimum 10000. 
    :returns: Tuple containing 4 values: 
              
              1. The confidence socres for each model, the best model should 
                 have the highest confidence
              2. The number of times each model was evaluated as a proportion of 
                 the number of evaluations
              3. The total number of model evaluations
              4. The scores that each model generated when evaluated. 
              
             :NOTE: That if the logit transform is True then the last item in 
                     the tuple would be scores that have been transformed by 
                     the logit function.
    :raises ValueError: If the ``p_value`` is not between 0 and 1.
    '''
    if p_value < 0.0 or p_value > 1.0:
        raise ValueError('The p value has to be between 0 and 1 and '
                         f'not: {p_value}')

    num_models = len(model_functions)
    total_samples = samples * num_models

    # initialize data storage
    # (use lists because will be of different lengths)
    evaluations = [[] for _ in range(num_models)]
    est_means = np.zeros(num_models)
    est_variances = np.zeros(num_models)
    # Number of times each model has been evaluated.
    eval_counts = np.zeros(num_models)

    init_num_evals = 3
    #start by evaluating each model 3 times
    for model_index, model_function in enumerate(model_functions):
        for _ in range(0, init_num_evals):
            train, test = split_function(data)
            score = model_function(train, test)
            if logit_transform:
                score = logit(score)
            evaluations[model_index].append(score)
        est_means[model_index] = np.mean(evaluations[model_index])
        est_variances[model_index] = np.var(evaluations[model_index], ddof=0)
        eval_counts[model_index] = len(evaluations[model_index])
    
    #initialize belief about location of best arm 
    pi = belief_calc(est_means, est_variances, eval_counts, total_samples)
    #run TTTS until hit required confidence
    #count number of evals
    num = init_num_evals * num_models
    #store running counts of each arm pulled
    #props = []
    #pis = []
    while max(pi) < 1 - p_value:
        # Get new train test split
        train, test = split_function(data)
        #pis.append(pi)
        #sample m-1
        m_1 = np.random.choice(range(0, num_models), 1, p=pi)[0]
        r = np.random.uniform(0, 1)
        if r<=0.5:
            #eval model m_1
            score = model_functions[m_1](train, test)
            if logit_transform:
                score = logit(score)
            evaluations[m_1].append(score)
            #update summary stats
            est_means[m_1] = np.mean(evaluations[m_1])
            est_variances[m_1] = np.var(evaluations[m_1],ddof=0)
            eval_counts[m_1] += 1 
            logger.info("Evalaution: %s, Model: %s", str(num), str(m_1))
        else:
            #sample other model
            m_2 = np.random.choice(range(0, num_models), 1, p=pi)[0]
            #resample until unique from model 1
            while m_1==m_2:
                m_2 = np.random.choice(range(0, num_models), 1, p=pi)[0]
            #eval m_2
            score = model_functions[m_2](train, test)
            if logit_transform:
                score = logit(score)
            evaluations[m_2].append(score)
            #update summary stats
            est_means[m_2] = np.mean(evaluations[m_2])
            est_variances[m_2] = np.var(evaluations[m_2],ddof=0)
            eval_counts[m_2] += 1  
            logger.info("Evalaution: %s, Model: %s", str(num), str(m_2))
        num += 1
        #update belief
        pi = belief_calc(est_means,est_variances,eval_counts, total_samples)
        logger.info("Evalaution: %s, Model confidences: %s", str(num), str(pi))
    logger.info("selected model %s", str(np.argmax(pi)))
    props = [x / sum(eval_counts) for x in eval_counts]
    return pi, props, num, evaluations

def sequential_halving(data: List[Dict[str, Any]], 
                       model_functions: List[Callable[[List[Dict[str, Any]],
                                                       List[Dict[str, Any]]], float]],
                       split_function: Callable[[List[Dict[str, Any]]], 
                                                Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
                       budget: int, logit_transform: bool = False 
                       ) -> Tuple[int, List[float], List[List[float]]]:
    '''
    :param data: A list of dictionaries, that as a whole represents the entire 
                 dataset. Each dictionary within the list represents 
                 one sample from the dataset.
    :param model_functions: A list of functions that represent different 
                            models e.g. pytorch model. Which take a train and  
                            test dataset as input and returns a metric score 
                            e.g. Accuracy. The model functions should not have 
                            random seeds set else it defeats the point of 
                            finding the best model independent of the random 
                            seed and data split.
    :param split_function: A function that can be used to split the data into 
                           train and test splits. This should produce random 
                           splits each time it is called. If you would like to 
                           use a fixed split each time, you can hard code this 
                           function to produce a fixed split each time.
    :param budget: The total number of evaluations 
    :param logit_transform: Whether to transform the model function's returned 
                            metric score by the logit function.
    :returns: Tuple containing 3 values: 
    
              1. The best performing model function index given the budget
              2. The number of times each model was evaluated as a proportion 
                 of the total number of evaluations
              3. The scores that each model generated when evaluated. 
              
              :NOTE: That if the logit transform is True then the last item in 
                     the tuple would be scores that have been transformed by 
                     the logit function.
    :raises ValueError: Given budget :math:`T` and the models :math:`N` this 
                        will be raised if: :math:`T < (|N| * \log_2|N|)`
    '''
    num_models = len(model_functions)
    R = math.ceil(np.log2(num_models))
    min_num_models = num_models * R
    if budget < min_num_models:
        raise ValueError(f'The budget {budget} cannot be smaller than (the '
                         f'number of models to evaluate  {num_models}) * '
                         '(the log to the base 2 of the number of models '
                         f'{R}) = {min_num_models}')
    evaluations = [[] for x in range(0, num_models)]
    candidate_names = [x for x in range(0, num_models)]
    candidate_est_means = [0 for x in range(0, num_models)]
    _round = 0
    while len(candidate_names) != 1:
        # calc number of evals for this round
        number_candidates = len(candidate_names)
        num_evals = math.floor(budget / (number_candidates * R))
        logger.info("Starting round %s, with %s models, with %s this round", 
                    str(_round), str(number_candidates), str(num_evals))
        # collect evaluations
        for candidate_name in candidate_names:
            for _ in range(0, num_evals):
                train, test = split_function(data)
                score = model_functions[candidate_name](train, test)
                if logit_transform:
                    score = logit(score)
                evaluations[candidate_name].append(score)
            # update means
            candidate_index = candidate_names.index(candidate_name)
            candidate_est_mean = np.mean(evaluations[candidate_name])
            candidate_est_means[candidate_index] = candidate_est_mean
        # remove approx half models
        for _ in range(0, math.floor(number_candidates / 2)):
            drop_index = np.argmin(candidate_est_means)
            del candidate_names[drop_index]
            del candidate_est_means[drop_index]
        _round = _round + 1
    total_num_evals = sum([len(model_evaluations) for model_evaluations in evaluations])
    props = [len(model_evaluations) / total_num_evals for model_evaluations in evaluations]
    return candidate_names[0], props, evaluations 

def non_adaptive_fb(data: List[Dict[str, Any]], 
                    model_functions: List[Callable[[List[Dict[str, Any]],
                                                    List[Dict[str, Any]]], float]],
                    split_function: Callable[[List[Dict[str, Any]]], 
                                             Tuple[List[Dict[str, Any]], 
                                                   List[Dict[str, Any]]]],
                    budget: int, logit_transform: bool = False 
                    ) -> Tuple[int, List[List[float]]]:
    '''
    :param data: A list of dictionaries, that as a whole represents the entire 
                 dataset. Each dictionary within the list represents 
                 one sample from the dataset.
    :param model_functions: A list of functions that represent different 
                            models e.g. pytorch model. Which take a train and  
                            test dataset as input and returns a metric score 
                            e.g. Accuracy. The model functions should not have 
                            random seeds set else it defeats the point of 
                            finding the best model independent of the random 
                            seed and data split.
    :param split_function: A function that can be used to split the data into 
                           train and test splits. This should produce random 
                           splits each time it is called. If you would like to 
                           use a fixed split each time, you can hard code this 
                           function to produce a fixed split each time.
    :param budget: The total number of evaluations 
    :param logit_transform: Whether to transform the model function's returned 
                            metric score by the logit function.
    :returns: Tuple containing 2 values: 
              
              1. The best performing model function index given the budget
              2. The scores that each model generated when evaluated. 
              
              :NOTE: That if the logit transform is True then the last item in 
                     the tuple would be scores that have been transformed by 
                     the logit function.
    :raises ValueError: Given budget :math:`T` and the models :math:`N` this 
                        will be raised if: :math:`T < |N|`
    '''
    num_models = len(model_functions)
    if budget < num_models:
        raise ValueError(f'The budget {budget} cannot be smaller than the '
                         f'number of models to evaluate {num_models}')

    evals: List[List[float]] = []
    num_evals_per_model = math.floor(budget / num_models)
    for model_function in model_functions:
        model_evals: List[float] = []
        for _ in range(0, num_evals_per_model):
            train, test = split_function(data)
            score = model_function(train, test)
            if logit_transform:
                score = logit(score)
            model_evals.append(score)
        evals.append(model_evals)
    # return model index with largest sample mean, and all the models scores.
    eval_means = np.mean(evals, axis=1)
    num_means = len(eval_means)
    assert_msg = f'The number of means: {num_means} should equal the number of'\
                 f' models {num_models}'
    assert len(eval_means) == num_models, assert_msg
    return np.argmax(eval_means), evals

def non_adaptive_fc(data: List[Dict[str, Any]], 
                    model_functions: List[Callable[[List[Dict[str, Any]],
                                                    List[Dict[str, Any]]], float]],
                    split_function: Callable[[List[Dict[str, Any]]], 
                                             Tuple[List[Dict[str, Any]], 
                                                   List[Dict[str, Any]]]],
                    p_value: float, logit_transform: bool = False, 
                    samples: int = 100000
                    ) -> Tuple[List[float], List[float], int, List[List[float]]]:
    '''
    :param data: A list of dictionaries, that as a whole represents the entire 
                 dataset. Each dictionary within the list represents 
                 one sample from the dataset.
    :param model_functions: A list of functions that represent different 
                            models e.g. pytorch model. Which take a train and  
                            test dataset as input and returns a metric score 
                            e.g. Accuracy. The model functions should not have 
                            random seeds set else it defeats the point of 
                            finding the best model independent of the random 
                            seed and data split.
    :param split_function: A function that can be used to split the data into 
                           train and test splits. This should produce random 
                           splits each time it is called. If you would like to 
                           use a fixed split each time, you can hard code this 
                           function to produce a fixed split each time.
    :param p_value: The significance value for the best model to be truely the 
                    best model e.g. 0.05 if you want to be at 
                    least 95% confident.
    :param logit_transform: Whether to transform the model function's returned 
                            metric score by the logit function.
    :param samples: Number of samples to generate from our belief distribution 
                    for each model. This argument is passed directly to 
                    :func:`fiesta.util.belief_calc` within this function. 
                    This should be large e.g. minimum 10000. 
    :returns: Tuple containing 4 values: 
              
              1. The confidence socres for each model, the best model should 
                 have the highest confidence
              2. The number of times each model was evaluated as a proportion of 
                 the number of evaluations
              3. The total number of model evaluations
              4. The scores that each model generated when evaluated. 
              
              :NOTE: That if the logit transform is True then the last item in 
                     the tuple would be scores that have been transformed by 
                     the logit function.
    :raises ValueError: If the ``p_value`` is not between 0 and 1.
    '''
    if p_value < 0.0 or p_value > 1.0:
        raise ValueError('The p value has to be between 0 and 1 and '
                         f'not: {p_value}')

    num_models = len(model_functions)
    total_samples = samples * num_models

    evaluations = [[] for i in range(num_models)]
    est_means = np.zeros(num_models)
    est_variances = np.zeros(num_models)
    eval_counts = np.zeros(num_models)

    # start by evaluating each model 3 times
    for model_index, model_function in enumerate(model_functions):
        for _ in range(0, 3):
            train, test = split_function(data)
            score = model_function(train, test)
            if logit_transform:
                score = logit(score)
            evaluations[model_index].append(score)
        est_means[model_index] = np.mean(evaluations[model_index])
        est_variances[model_index] = np.var(evaluations[model_index], ddof=0)
        eval_counts[model_index] = len(evaluations[model_index])
    
    #initialize belief about location of best arm 
    pi = belief_calc(est_means, est_variances, eval_counts, total_samples)
    
    # count number of evals
    num = 3 * num_models
    # run until hit required confidence
    while max(pi) < 1 - p_value:
        # sample all arms
        for model_index, model_function in enumerate(model_functions):
            train, test = split_function(data)
            score = model_function(train, test)
            if logit_transform:
                score = logit(score)
            evaluations[model_index].append(score)
            # update summary stats
            est_means[model_index] = np.mean(evaluations[model_index])
            est_variances[model_index] = np.var(evaluations[model_index], ddof=0)
            eval_counts[model_index] += 1 
        num += num_models
        # update belief
        pi = belief_calc(est_means, est_variances, eval_counts, total_samples)
        logger.info("Evalaution: %s, Model confidences: %s", str(num), str(pi))
    logger.info("selected model %s", str(np.argmax(pi)))
    props = [x / sum(eval_counts) for x in eval_counts]
    return pi, props, num, evaluations