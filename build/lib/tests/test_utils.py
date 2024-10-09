# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from slim_gsgp.utils.utils import get_best_max, get_best_min
import random

def test_get_best_max():
    class IndivTest:
        def __init__(self, fitness):
            self.fitness = fitness

        def __eq__(self, other):
            return self.fitness == other.fitness

    class PopTest:
        def __init__(self, population):
            self.population = population
            self.fit = [indiv.fitness for indiv in self.population]

    example1 =  IndivTest(1)
    example2 =  IndivTest(2)
    example3 =  IndivTest(3)
    example4 =  IndivTest(4)
    example5 =  IndivTest(5)

    example_list = [example1, example2, example3, example4, example5]
    expected_top = sorted(example_list, key=lambda x: x.fitness, reverse=True)[:3]

    for i in range(30):
        random.shuffle(example_list)
        example_pop = PopTest(example_list)
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
