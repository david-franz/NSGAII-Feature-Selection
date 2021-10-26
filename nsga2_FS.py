import sys
import random

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling

from dataloader import DataLoader

NUMBER_OF_FEATURES = 167 # this is for the musk data set
NUMBER_OF_CLASSES = 2 # this is for the musk data set

# design is a binary string
def fitness(binary_string):
	return wrapper_based_fitness_function(binary_string), (count_1s(binary_string)/NUMBER_OF_FEATURES)

class ProblemWrapper(Problem):

	def _evaluate(self, designs, out, *args, **kwargs):
		#for i in designs:
			#print(i.astype(int))

		res = list()
		for design in designs:
			res.append(fitness(''.join([str(i) for i in design.astype(int)]))) # need to define this fitness function
		
		out['F'] = np.array(res) # F is for function output

def convert_df_to_list(data):
	data_as_list = list()
	for key in data.keys():
		vector = list()
		for feature in data[key]:
			vector.append(feature)
		data_as_list.append(vector)

	return data_as_list

def wrapper_based_fitness_function(binary_string):
	if binary_string == ('0' * len(binary_string)):
		return 0

	filter_data = lambda data_as_list: [[data_as_list[j][i] for i in range(len(data_as_list[0])-1)] for j in range(len(data_as_list))]
	get_class_labels = lambda data_as_list: [1 if data_as_list[i][len(data_as_list[0])-1]=="musk" else 0 for i in range(len(data_as_list))]

	training_df_as_list = convert_df_to_list(training_data.T)

	training_data_filtered = filter_data(training_df_as_list)
	training_class_labels = get_class_labels(training_df_as_list)

	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(training_data_filtered, training_class_labels)

	testing_df_as_list = convert_df_to_list(testing_data.T)

	testing_data_filtered = filter_data(testing_df_as_list)
	testing_class_labels = get_class_labels(testing_df_as_list)

	predicted_class_labels = knn.predict(testing_data_filtered)

	count_correct = 0
	for i in range(len(predicted_class_labels)):
		if predicted_class_labels[i] == testing_class_labels[i]:
			count_correct += 1

	return float(count_correct) / len(predicted_class_labels)

def count_1s(binary_string):
	return sum([int(i) for i in list(binary_string)])

if __name__ == '__main__':

	df = DataLoader.load_data("clean1.data")
	
	seed = random.randint(0,100)

	training_data, testing_data = train_test_split(df.T, test_size=0.2, random_state=seed)
	

	bs = ''.join([str(random.randint(0,1)) for i in range(NUMBER_OF_FEATURES)])

	problem = ProblemWrapper(n_var=167, n_obj=2, xl=0, xu=1, type_var=bool) # xl is lower bound, xu is upper bound

	algorithm = NSGA2(pop_size=50, sampling=get_sampling("bin_random"), crossover=get_crossover("bin_hux"), mutation=get_mutation("bin_bitflip"), eliminate_duplicates=True)

	stop_criteria = ('n_gen', 25) # number_of_generations=100 is our stopping criteria

	results = minimize(problem=problem, algorithm=algorithm, termination=stop_criteria)

	fitnesses = list()
	for result in results.X.astype(int):
		result = ''.join([str(i) for i in result])

		print(result)
		print(count_1s(result))

		fitnesses.append(wrapper_based_fitness_function(result))

	print(fitnesses)


