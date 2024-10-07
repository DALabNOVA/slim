from slim.utils.utils import get_best_max, get_best_min
import random

def test_get_best_max():
    class indiv_test:
        def __init__(self, fitness):
            self.fitness = fitness

        def __eq__(self, other):
            return self.fitness == other.fitness

    class pop_test:
        def __init__(self, population):
            self.population = population
            self.fit = [indiv.fitness for indiv in self.population]

    example1 = indiv_test(1)
    example2 = indiv_test(2)
    example3 = indiv_test(3)
    example4 = indiv_test(4)
    example5 = indiv_test(5)

    example_list = [example1, example2, example3, example4, example5]
    expected_top = sorted(example_list, key=lambda x: x.fitness, reverse=True)[:3]

    for i in range(30):
        random.shuffle(example_list)
        example_pop = pop_test(example_list)
        result1, result2 = get_best_max(example_pop, 3)

        assert (example4 in expected_top and example5 in expected_top and
                result2 == example5)

def test_get_best_min():
    class indiv_test:
        def __init__(self, fitness):
            self.fitness = fitness

        def __eq__(self, other):
            return self.fitness == other.fitness

    class pop_test:
        def __init__(self, population):
            self.population = population
            self.fit = [indiv.fitness for indiv in self.population]

    example1 = indiv_test(1)
    example2 = indiv_test(2)
    example3 = indiv_test(3)
    example4 = indiv_test(4)
    example5 = indiv_test(5)

    example_list = [example1, example2, example3, example4, example5]

    for i in range(30):
        random.shuffle(example_list)
        example_pop = pop_test(example_list)
        result1, result2 = get_best_min(example_pop, 3)

        assert (example1 in result1 and example2 in result1 and example3 in result1 and
                result2 == example1)
