import math
from typing import List

import pytest
import numpy as np

from fiesta_test import model_generator, split_data
from fiesta.util import pull_arm, belief_calc, lists_same_size, fc_func_stats
from fiesta.util import fb_func_stats

@pytest.mark.parametrize("mean", (2, 3, 10, 0.7))
@pytest.mark.parametrize("sd", (0.2, 0.31, 1.4))
def test_pull_arm(mean, sd):
    values: List[float] = []
    for _ in range(0, 10):
        values.append(pull_arm(mean, sd))

    # the values from a normal distribution should be within 3 standard 
    # deviations away 99.7% of the time.
    not_in_limit = 0
    sd_3 = 3 * sd
    upper_limit = mean + sd_3
    lower_limit = mean - sd_3
    for value in values:
        if value > upper_limit or value < lower_limit:
            print(value)
            print(upper_limit)
            print(lower_limit)
            print(mean)
            print(sd)
            print(sd_3)
            not_in_limit += 1
    assert not_in_limit < 2


def test_belief_calc():
    # Case where there are 3 very distinct models
    est_means = np.array([10, 3, 8])
    est_variances = np.array([1.7, 0.1, 1.2])
    eval_counts = np.array([10, 6, 12])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert pi_values[0] > pi_values[1]
    assert pi_values[2] > pi_values[1]
    assert pi_values[0] > pi_values[2] 
    # Should return in essence a probabilty of the most confident model
    assert np.isclose(sum(pi_values), 1)

    # Case where there are 3 very distinct models
    est_means = np.array([9, 10, 8])
    est_variances = np.array([0.9, 1.2, 1.0])
    eval_counts = np.array([10, 6, 12])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert pi_values[1] > pi_values[0]
    assert pi_values[1] > pi_values[2]
    assert pi_values[0] > pi_values[2] 
    assert np.isclose(sum(pi_values), 1)

    # Case of just one winner
    est_means = np.array([10, 2])
    est_variances = np.array([0.9, 0.8])
    eval_counts = np.array([10, 6])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert pi_values[0] > pi_values[1] 
    assert np.isclose(sum(pi_values), 1)

    # Should work with only one but defeats the point of model evaluation
    est_means = np.array([10])
    est_variances = np.array([0.9])
    eval_counts = np.array([6])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert len(pi_values) == 1
    assert np.isclose(sum(pi_values), 1)

    # Should raise an error if any of the eval_counts are less than 3
    est_means = np.array([9, 10, 8])
    est_variances = np.array([0.9, 1.2, 1.0])
    eval_counts = np.array([3, 3, 2])
    with pytest.raises(ValueError):
        belief_calc(est_means, est_variances, eval_counts, samples)
    eval_counts = np.array([2, 2, 2])
    with pytest.raises(ValueError):
        belief_calc(est_means, est_variances, eval_counts, samples)
    eval_counts = np.array([1, 2, 2])
    with pytest.raises(ValueError):
        belief_calc(est_means, est_variances, eval_counts, samples)
    eval_counts = np.array([3, 3, 3])
    belief_calc(est_means, est_variances, eval_counts, samples)

    # Test that the default samples value works
    est_means = np.array([9, 10, 8])
    est_variances = np.array([0.9, 1.2, 1.0])
    eval_counts = np.array([10, 6, 12])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts)
    assert pi_values[1] > pi_values[0]
    assert pi_values[1] > pi_values[2]
    assert pi_values[0] > pi_values[2]
    assert np.isclose(sum(pi_values), 1)

def test_lists_same_size():
    # one list example
    list_0 = [1, 2, 3]
    lists_same_size(list_0)

    # three list example that should pass
    list_1 = ['another', 'day', 'today']
    list_2 = [1.0, 5.2, {'another': 1}]
    lists_same_size(list_0, list_1, list_2)

    # Case when the lists are not the same size
    list_3 = [1]
    with pytest.raises(ValueError):
        lists_same_size(list_0, list_2, list_3)
    
    # Empty list
    lists_same_size([])
    # Empty lists
    lists_same_size([], [])

@pytest.mark.parametrize("fc_func_name", ('TTTS', 'non_adaptive_fc', 'fc'))
def test_fc_func_stats(fc_func_name: str):
    N = 10
    correct_model_index = 2
    if fc_func_name == 'fc':
        with pytest.raises(ValueError):
            fc_func_stats(N=N, correct_model_index=correct_model_index,
                          fc_func_name=fc_func_name)
    else:
        train_test_data = list(np.random.normal(0.5, 0.01, 500))
        train_test_json = [{'x': sample} for sample in train_test_data]
        # model 2 > model 1 > model 0
        model_0 = model_generator(0.17, 0.02)
        model_1 = model_generator(0.27, 0.018)
        model_2 = model_generator(0.3, 0.015)
        models = [model_0, model_1, model_2]
        p_value = 0.2
        summary_stats = fc_func_stats(N=N, 
                                      correct_model_index=correct_model_index,
                                      fc_func_name=fc_func_name, 
                                      data=train_test_json, 
                                      model_functions=models, 
                                      split_function=split_data, p_value=p_value)
        _min, _mean, _max, perecent_correct = summary_stats
        
        assert _min < _max
        assert _min < _mean
        assert _max > _mean
        assert perecent_correct > 0.8

@pytest.mark.parametrize("fb_func_name", ('sequential_halving', 
                                          'non_adaptive_fb', 'fb'))
def test_fb_func_stats(fb_func_name: str):
    N = 100
    correct_model_index = 2
    if fb_func_name == 'fb':
        with pytest.raises(ValueError):
            fb_func_stats(N=N, correct_model_index=correct_model_index,
                          fb_func_name=fb_func_name)
    else:
        train_test_data = list(np.random.normal(0.5, 0.01, 500))
        train_test_json = [{'x': sample} for sample in train_test_data]
        # model 2 > model 1 > model 0
        model_0 = model_generator(0.17, 0.02)
        model_1 = model_generator(0.27, 0.028)
        model_2 = model_generator(0.3, 0.015)
        models = [model_0, model_1, model_2]
        budget = 40
        prob_correct = fb_func_stats(N=N, 
                                     correct_model_index=correct_model_index,
                                     fb_func_name=fb_func_name, 
                                     data=train_test_json, 
                                     model_functions=models, 
                                     split_function=split_data, budget=budget)
        assert prob_correct >= 0.0 and prob_correct <= 1.0