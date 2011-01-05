import uniform
import itertools


def fitness(x):
    return -x * x

parameters = {'x': [-100, 100, .1]}
max_fitness = None
max_params = None
for x, y in uniform.uniform_grid(fitness, parameters):
    if not max_fitness or x > max_fitness:
        max_fitness = x
        max_params = y
print((max_fitness, max_params))

max_fitness = None
max_params = None
for x, y in uniform.uniform_random(fitness, parameters):
    if not max_fitness or x > max_fitness:
        max_fitness = x
        max_params = y
print((max_fitness, max_params))

max_fitness = None
max_params = None
for x, y in itertools.islice(uniform.uniform_random_sample(fitness, parameters), 1000000):
    if not max_fitness or x > max_fitness:
        max_fitness = x
        max_params = y
print((max_fitness, max_params))
