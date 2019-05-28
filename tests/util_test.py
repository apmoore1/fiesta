import math
from typing import List

import pytest

from fiesta.util import pull_arm

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
