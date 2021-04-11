import statistics
from operator import mul
from functools import reduce

def product_aggregation(x):
    return reduce(mul, x, 1.0)

def sum_aggregation(x):
    return sum(x)

def max_aggregation(x):
    return max(x)

def min_aggregation(x):
    return min(x)

def maxabs_aggregation(x):
    return max(x, key=abs)

def median_aggregation(x):
    return statistics.median(x)

def mean_aggregation(x):
    return statistics.mean(x)


class Aggregations(object):

    def __init__(self):
        self.functions = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        return self.functions.get(name)

    def __getitem__(self, index):
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions