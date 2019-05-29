'''
Module that contains the main fiesta functions.

Functions:

TTTS - Top-Two Thompson Sampling
'''
from typing import List, Dict, Any, Tuple, Callable

import numpy as np

from fiesta.util import belief_calc

def TTTS(data: List[Dict[str, Any]], 
         model_functions: List[Callable[[Tuple[List[Dict[str, Any]]],
                                               List[Dict[str, Any]]], float]],
         split_function: Callable[[List[Dict[str, Any]]], 
                                  Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
         confidence_level: float,
         samples: int = 100000
         ) -> Tuple[float, List[float], List[float], int]:
    '''
    This function requires as input at least some sort of List of method functions
    or perhaps more selectively a List of AllenNLP config path, then secondly 
    a data soucre of which we should allow them to give it optionally as a 
    train test split or not then lastly wether or not to 
    randomly split the data.

    Whether the model requires a development split is up to the method 
    functions

    :param data: A list of dictionaries, that as a whole represents the entire 
                 dataset. Each dictionary should within the list represents 
                 one sample from the dataset.
    :param model_functions: A list of functions that represent different 
                            models e.g. pytorch model. Which take a dataset 
                            as input and returns a metric score e.g. Accuracy.
                            The model functions should not have random seeds 
                            set else it defeats the point of finding the best 
                            model independent of the random seed and data split.
    :param split_function: A function that can be used to split the data into 
                           train and test splits. This should produce random 
                           splits each time it is called. If you would like 
                           to not use a random split and this not find the 
                           best the model independent of the random seed and 
                           data split, you can hard code this function to 
                           produce a set split each time.
    :param confidence_level: The level of confidence required that the 
                             best model is truly the best model e.g. 0.95 if 
                             you want a p value of 0.05. Confidence level is a
                             re-wording of delta that is used in the original 
                             algorthim.
    :param samples: Number of samples to generate from our belief distribution 
                    for each model. This argument is passed directly to 
                    `fiesta.util.belief_calc` within this function. 
                    This should be large e.g. minimum 10000. 
    :returns: Tuple of numbers
    :raises ValueError: If the confidence level is not between 0 and 1.
    '''
    if confidence_level < 0.0 or confidence_level > 1.0:
        raise ValueError('The confidence level has to be between 0 and 1 and '
                         f'not: {confidence_level}')

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
    props = []
    pis = []
    while max(pi) < 1 - confidence_level:
        # Get new train test split
        train, test = split_function(data)

        props.append([x / sum(eval_counts) for x in eval_counts])
        pis.append(pi)
        #sample m-1
        m_1 = np.random.choice(range(0, num_models), 1, p=pi)[0]
        r = np.random.uniform(0, 1)
        if r<=0.5:
            #eval model m_1
            score = model_functions[m_1](train, test)
            evaluations[m_1].append(score)
            #update summary stats
            est_means[m_1] = np.mean(evaluations[m_1])
            est_variances[m_1] = np.var(evaluations[m_1],ddof=0)
            eval_counts[m_1] += 1 
            print(f"evalaution: {str(num)}, model: {str(m_1)}")
        else:
            #sample other model
            m_2 = np.random.choice(range(0, num_models), 1, p=pi)[0]
            #resample until unique from model 1
            while m_1==m_2:
                m_2 = np.random.choice(range(0, num_models), 1, p=pi)[0]
            #eval m_2
            score = model_functions[m_2](train, test)
            evaluations[m_2].append(score)
            #update summary stats
            est_means[m_2] = np.mean(evaluations[m_2])
            est_variances[m_2] = np.var(evaluations[m_2],ddof=0)
            eval_counts[m_2] += 1  
            print(f"evalaution: {str(num)}, model: {str(m_2)}")
        num += 1
        #update belief
        pi = belief_calc(est_means,est_variances,eval_counts, total_samples)
        print(pi)
    print(f"selected model {str(np.argmax(pi))}")
    return pi, props,pis,num