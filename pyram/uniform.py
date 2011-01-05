import numpy as np
import itertools
import random


def uniform_grid(fitness_func, parameters):
    """Uniform parameter optimization that checks all possible values

    Values are visited in a grid order

    Args:
        fitness_func: Fitness function that takes keyword arguments whos values
            are keys in 'parameters'.  Each keyword argument takes a float.
            The fitness function returns a float that we seek to maximize.
        parameters: Dict with keys as parameter names and values as
            (low, high, resolution) where generated parameters are [low, high)
            and resolution is a hint at the relevant scale of the parameter.

    Yields:
        Iterator of (fitness, params) where
        fitness: The value returned by the fitness_func given params
        params: Dict whos keys are those in parameters and values are floats
    """
    ranges = [np.arange(*x) for x in parameters.values()]
    for param_values in itertools.product(*ranges):
        params = dict(zip(parameters, param_values))
        yield fitness_func(**params), params


def uniform_random(fitness_func, parameters):
    """Uniform parameter optimization that checks all possible values

    Values are visited in random order

    Args:
        fitness_func: Fitness function that takes keyword arguments whos values
            are keys in 'parameters'.  Each keyword argument takes a float.
            The fitness function returns a float that we seek to maximize.
        parameters: Dict with keys as parameter names and values as
            (low, high, resolution) where generated parameters are [low, high)
            and resolution is a hint at the relevant scale of the parameter.

    Yields:
        Iterator of (fitness, params) where
        fitness: The value returned by the fitness_func given params
        params: Dict whos keys are those in parameters and values are floats
    """
    ranges = [np.arange(*x) for x in parameters.values()]
    for x in ranges:
        random.shuffle(x)
    for param_values in itertools.product(*ranges):
        params = dict(zip(parameters, param_values))
        yield fitness_func(**params), params


def uniform_random_sample(fitness_func, parameters):
    """Uniform parameter optimization that samples values

    Values are visited in random order, may be revisited, and this never
    terminates.

    Args:
        fitness_func: Fitness function that takes keyword arguments whos values
            are keys in 'parameters'.  Each keyword argument takes a float.
            The fitness function returns a float that we seek to maximize.
        parameters: Dict with keys as parameter names and values as
            (low, high, resolution) where generated parameters are [low, high)
            and resolution is a hint at the relevant scale of the parameter.

    Yields:
        Iterator of (fitness, params) where
        fitness: The value returned by the fitness_func given params
        params: Dict whos keys are those in parameters and values are floats
    """
    shift_scales = [(low, high - low)
                    for low, high, res in parameters.values()]
    while 1:
        params = dict((name, x + random.random() * y)
                      for name, (x, y) in zip(parameters, shift_scales))
        yield fitness_func(**params), params
