from typing import List, Dict, Any, Tuple, Callable
import statistics

import numpy as np
from flaky import flaky
import pytest
from scipy.special import expit

from fiesta import TTTS, sequential_halving, non_adaptive_fb, non_adaptive_fc

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

# Fixed Budget tests
@flaky
@pytest.mark.parametrize("function_name", ('sequential_halving', 'non_adaptive_fb'))
@pytest.mark.parametrize("logit_transform", (True, False))
def test_fixed_budget_methods(logit_transform: bool, function_name: str):
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]
    # model 2 > model 1 > model 0
    model_0 = model_generator(0.17, 0.03)
    model_1 = model_generator(0.1, 0.02)
    model_2 = model_generator(0.15, 0.05)
    model_3 = model_generator(0.3, 0.02)
    model_4 = model_generator(0.27, 0.02)
    models = [model_0, model_1, model_2, model_3, model_4]

    budget = 25
    fb_function_mapper = {'sequential_halving': sequential_halving, 
                          'non_adaptive_fb': non_adaptive_fb}
    fb_function = fb_function_mapper[function_name]
    func_return = fb_function(train_test_json, models, split_data, budget,
                              logit_transform=logit_transform)
    if function_name == 'sequential_halving':
        best_model, model_prop, models_evaluations = func_return
        assert np.isclose(sum(model_prop), 1.0)
    elif function_name == 'non_adaptive_fb':
        best_model, models_evaluations = func_return
    else:
        raise ValueError(f'Do not recognise this function name {function_name}')
    assert best_model == 3
    total_evaluations = sum([len(model_evaluations) for model_evaluations in models_evaluations])
    assert total_evaluations <= budget

    for model_evaluations in models_evaluations:
        for score in model_evaluations:
            if logit_transform:
                score = expit(score)
            assert score >= 0
            assert score <= 1

# Fixed constriant tests

def all_same(props: List[float]) -> bool:
    '''
    Returns True if all of the items in the list are the same. Reference:
    https://stackoverflow.com/questions/3787908/python-determine-if-all-items-of-a-list-are-the-same-item
    '''
    assert len(props) > 0
    return all(prop == props[0] for prop in props)

@flaky(max_runs=5, min_passes=4)
@pytest.mark.parametrize("function_name", ('TTTS', 'non_adaptive_fc'))
def test_fc_confidence_scores(function_name: str):
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]
    # model 2 > model 1 > model 0
    model_0 = model_generator(0.17, 0.03)
    model_1 = model_generator(0.27, 0.02)
    model_2 = model_generator(0.3, 0.02)
    models = [model_0, model_1, model_2]

    fc_function_mapper = {'TTTS': TTTS, 
                          'non_adaptive_fc': non_adaptive_fc}
    fc_function = fc_function_mapper[function_name]

    model_confidence_scores, model_prop, total_evals, _ = fc_function(train_test_json, models, 
                                                               split_data, 0.05)
    # If less than 10 then it never did any thompson sampling
    assert total_evals > 10
    # Model proportion should be a probability
    assert np.isclose(sum(model_prop), 1.0)
    assert sum([prop * total_evals for prop in model_prop]) == total_evals
    assert np.argmax(model_confidence_scores) == 2
    # The best model should have the most runs in the TTTS in the non adaptive 
    # the number of runs should be equal
    if function_name == 'TTTS':
        assert np.argmax(model_prop) != 0
        assert not all_same(model_prop)
    elif function_name == 'non_adaptive_fc':
        assert all_same(model_prop)


    models = [model_1, model_2, model_0]
    model_confidence_scores, model_prop, total_evals, _ = fc_function(train_test_json, models, split_data, 0.05)
    assert total_evals > 10
    assert np.argmax(model_confidence_scores) == 1
    # The best model should have the most runs in the TTTS in the non adaptive 
    # the number of runs should be equal
    if function_name == 'TTTS':
        assert np.argmax(model_prop) != 2
        assert not all_same(model_prop)
    elif function_name == 'non_adaptive_fc':
        assert all_same(model_prop)

@flaky(max_runs=5, min_passes=3)
@pytest.mark.parametrize("logit_transform", (True, False))
@pytest.mark.parametrize("function_name", ('TTTS', 'non_adaptive_fc'))
def test_fc_num_runs(logit_transform: bool, function_name: str):
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]
    # with three very easy to distinguish models it should have very few 
    # number of evals compared to models that output similar scores
    model_0 = model_generator(0.11, 0.05)
    model_1 = model_generator(0.15, 0.01)
    model_2 = model_generator(0.2, 0.04)
    model_3 = model_generator(0.3, 0.03)
    models = [model_0, model_1, model_2, model_3]

    fc_function_mapper = {'TTTS': TTTS, 
                          'non_adaptive_fc': non_adaptive_fc}
    fc_function = fc_function_mapper[function_name]

    if function_name == 'TTTS':
        _, _ , total_easy_evals, _ = fc_function(train_test_json, models, split_data, 0.15, 
                                                 logit_transform=logit_transform)

        model_0 = model_generator(0.26, 0.03)
        model_1 = model_generator(0.28, 0.01)
        model_2 = model_generator(0.29, 0.03)
        model_3 = model_generator(0.3, 0.05)
        models = [model_0, model_1, model_2, model_3]

        _, _ , total_hard_evals, _ = fc_function(train_test_json, models, split_data, 0.15,
                                                 logit_transform=logit_transform)

        assert total_hard_evals > total_easy_evals
    elif function_name == 'non_adaptive_fc':
        model_0 = model_generator(0.26, 0.03)
        model_1 = model_generator(0.28, 0.01)
        model_2 = model_generator(0.29, 0.03)
        model_3 = model_generator(0.3, 0.05)
        models = [model_0, model_1, model_2, model_3]

        _, _ , total_hard_evals, _ = fc_function(train_test_json, models, split_data, 0.15,
                                                 logit_transform=logit_transform)

    # With a higher p value it should take fewer runs
    _, _ , total_smaller_p_evals, _ = fc_function(train_test_json, models, split_data, 0.5,
                                                  logit_transform=logit_transform)

    assert total_hard_evals > total_smaller_p_evals

@pytest.mark.parametrize("p_value", (-0.1, 1.1, 5.0, -100.0))
@pytest.mark.parametrize("function_name", ('TTTS', 'non_adaptive_fc'))
def test_fc_p_value(p_value: float, function_name: str):
    '''
    These are all the p values that are not acceptable.
    '''
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]

    model_0 = model_generator(0.11, 0.05)
    model_1 = model_generator(0.2, 0.04)
    model_2 = model_generator(0.3, 0.03)
    models = [model_0, model_1, model_2]

    fc_function_mapper = {'TTTS': TTTS, 
                          'non_adaptive_fc': non_adaptive_fc}
    fc_function = fc_function_mapper[function_name]

    with pytest.raises(ValueError):
        fc_function(train_test_json, models, split_data, p_value)

@pytest.mark.parametrize("logit_transform", (True, False))
@pytest.mark.parametrize("function_name", ('TTTS', 'non_adaptive_fc'))
def test_fc_evaluation_results(logit_transform: bool, function_name: str):
    train_test_data = list(np.random.normal(0.5, 0.01, 500))
    train_test_json = [{'x': sample} for sample in train_test_data]

    model_0 = model_generator(0.11, 0.05)
    model_1 = model_generator(0.2, 0.04)
    model_2 = model_generator(0.3, 0.03)
    models = [model_0, model_1, model_2]

    fc_function_mapper = {'TTTS': TTTS, 
                          'non_adaptive_fc': non_adaptive_fc}
    fc_function = fc_function_mapper[function_name]
    _, _, _, models_evaluations = fc_function(train_test_json, models, split_data, 0.2,
                                              logit_transform=logit_transform)
    
    for model_evaluations in models_evaluations:
        for score in model_evaluations:
            if logit_transform:
                score = expit(score)
            assert score >= 0
            assert score <= 1



    