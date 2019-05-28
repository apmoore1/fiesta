import math
from typing import List

import pytest
import numpy as np

from fiesta.util import pull_arm, belief_calc

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

    # Case where there are 3 very distinct models
    est_means = np.array([9, 10, 8])
    est_variances = np.array([0.9, 1.2, 1.0])
    eval_counts = np.array([10, 6, 12])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert pi_values[1] > pi_values[0]
    assert pi_values[1] > pi_values[2]
    assert pi_values[0] > pi_values[2] 

    # Case of just one winner
    est_means = np.array([10, 2])
    est_variances = np.array([0.9, 0.8])
    eval_counts = np.array([10, 6])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert pi_values[0] > pi_values[1] 

    # Should work with only one but defeats the point of model evaluation
    est_means = np.array([10])
    est_variances = np.array([0.9])
    eval_counts = np.array([6])
    samples = 10000
    pi_values = belief_calc(est_means, est_variances, eval_counts, samples)
    assert len(pi_values) == 1

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
