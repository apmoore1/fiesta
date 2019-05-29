from typing import List, Dict, Any, Tuple, Callable
import statistics

import numpy as np
from flaky import flaky
import pytest

from fiesta.fiesta import TTTS

def split_data(data: List[Dict[str, Any]]
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''
    :param data: The list of data
    :returns: The data split into 80/20 split as a Tuple.
    '''
    # 80% train data
    train_size = int((len(data) / 100) * 80)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def model_generator(mean: float, sd: float
                    ) -> Callable[[List[Dict[str, Any]], 
                                   List[Dict[str, Any]]], float]:
    '''
    :param mean: The mean of the normal distribution
    :param sd: The standard deviation of the normal distribution
    :returns: A function that takes as input a train and test data set and 
              returns essentially the a value from the normal distribution 
              generated from the mean and sd values given as arguments.
    '''
    def model_func(train: List[Dict[str, Any]], test: List[Dict[str, Any]]
                   ) -> float:
        all_x_values = []
        for value in train:
            all_x_values.append(value['x'])
        train_mean = statistics.mean(all_x_values)
        test_mean = train_mean
        train_mean += np.random.normal(mean, sd)
        if train_mean > 1.0 or train_mean < 0.0:
            raise ValueError('The train mean has to be between 0 and 1: '
                             f'{train_mean}')
        return abs(train_mean - test_mean)
    return model_func

@flaky(max_runs=5, min_passes=1)
def test_TTTS_confidence_scores():
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]
    # model 2 > model 1 > model 0
    model_0 = model_generator(0.18, 0.05)
    model_1 = model_generator(0.25, 0.04)
    model_2 = model_generator(0.3, 0.03)
    models = [model_0, model_1, model_2]

    model_confidence_scores, model_prop, total_evals = TTTS(train_test_json, models, 
                                                            split_data, 0.1)
    assert np.isclose(sum(model_prop), 1.0)
    assert sum([prop * total_evals for prop in model_prop]) == total_evals
    assert np.argmax(model_confidence_scores) == 2
    assert np.argsort(model_confidence_scores).tolist() == [0, 1, 2]

    models = [model_1, model_2, model_0]
    model_confidence_scores, _, _ = TTTS(train_test_json, models, split_data, 0.1)
    assert np.argmax(model_confidence_scores) == 1
    assert np.argsort(model_confidence_scores).tolist() == [2, 0, 1]

@flaky(max_runs=5, min_passes=1)
@pytest.mark.parametrize("logit_transform", (True, False))
def test_TTTS_num_runs(logit_transform: bool):
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]
    # with three very easy to distinguish models it should have very few 
    # number of evals compared to models that output similar scores
    model_0 = model_generator(0.11, 0.05)
    model_1 = model_generator(0.2, 0.04)
    model_2 = model_generator(0.3, 0.03)
    models = [model_0, model_1, model_2]

    _, _ , total_easy_evals = TTTS(train_test_json, models, split_data, 0.1, 
                                   logit_transform=logit_transform)

    model_0 = model_generator(0.26, 0.03)
    model_1 = model_generator(0.28, 0.01)
    model_2 = model_generator(0.3, 0.05)
    models = [model_0, model_1, model_2]

    _, _ , total_hard_evals = TTTS(train_test_json, models, split_data, 0.1,
                                   logit_transform=logit_transform)

    assert total_hard_evals > total_easy_evals

    # With a higher p value it should take fewer runs
    _, _ , total_smaller_p_evals = TTTS(train_test_json, models, split_data, 0.2,
                                        logit_transform=logit_transform)

    assert total_hard_evals > total_smaller_p_evals

@pytest.mark.parametrize("p_value", (-0.1, 1.1, 5.0, -100.0))
def test_TTTS_p_value(p_value: float):
    '''
    These are all the p values that are not acceptable.
    '''
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]

    model_0 = model_generator(0.11, 0.05)
    model_1 = model_generator(0.2, 0.04)
    model_2 = model_generator(0.3, 0.03)
    models = [model_0, model_1, model_2]

    with pytest.raises(ValueError):
        TTTS(train_test_json, models, split_data, p_value)

    