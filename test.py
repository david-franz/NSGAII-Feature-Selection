import sys
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

import autograd.numpy as anp


class Knapsack(Problem):
    def __init__(self,
                 n_items,  # number of items that can be picked up
                 W,  # weights for each item
                 P,  # profit of each item
                 C,  # maximum capacity
                 ):
        super().__init__(n_var=n_items, n_obj=1, n_constr=1, xl=0, xu=1, type_var=bool)

        self.W = W
        self.P = P
        self.C = C

    def _evaluate(self, x, out, *args, **kwargs):
        for i in x:
            print(i.astype(int))
        sys.exit(0)

        out_f = -anp.sum(self.P * x, axis=1)
        out_g = (anp.sum(self.W * x, axis=1) - self.C)

        out["F"] = out_f
        out["G"] = out_g


def create_random_knapsack_problem(n_items, seed=1):
    anp.random.seed(seed)

    W = anp.random.randint(1, 100, size=n_items)
    print(W)
    P = anp.random.randint(1, 100, size=n_items)
    print(P)
    C = int(anp.sum(W) / 10)
    print(C)

    return Knapsack(n_items, W, P, C)


problem = create_random_knapsack_problem(168)

algorithm = GA(
    pop_size=200,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_hux"),
    mutation=get_mutation("bin_bitflip"),
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               verbose=False)

print("Best solution found: %s" % res.X.astype(int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)