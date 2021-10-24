import numpy as np

from pymoo.optimise import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2

NUMBER_OF_FEATURES = 10 # will be changed

class ProblemWrapper(Problem):

	def _evaluate(self, designs, out, *args, **kwargs):
		res = list()
		for design in designs:
			res.append(fitness(design)) # need to define this function
		
		out['F'] = np.array(res) # F is for function output

problem = ProblemWrapper(n_var=2, n_obj=2, xl=[0.,0.], xu=[1.,1.]) # xl is lower bound, xu is upper bound

algorithm = NSGA2(pop_size=100)

stop_criteria = ('n_gen', 100) # number_of_generations=100 is our stopping criteria

results = minimize(problem=problem, algorithm=algorithm, termination=stop_criteria)

print(results.F) # function outputs of optimal solutions (size of population)
print(results.X) # function inputs

# need to define feature rate, error rate
# these will be the outputs of my function fitness

# need to implement wrapper based fitness function