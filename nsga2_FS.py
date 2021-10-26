import sys
import random

import numpy as np
import pandas as pd
import matplotlib as plt

from functools import cmp_to_key
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.indicators.hv import Hypervolume

from dataloader import DataLoader

def fitness(binary_string):
	return wrapper_based_fitness_function(binary_string), (count_1s(binary_string)/NUMBER_OF_FEATURES)

class ProblemWrapper(Problem):

	def _evaluate(self, designs, out, *args, **kwargs):
		res = list()
		for design in designs:
			
			bs = ''.join([str(i) for i in design.astype(int)])
			res.append(fitness(bs))

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
		return float('-inf')

	filter_data = lambda data_as_list: [[data_as_list[j][i] for i in range(len(data_as_list[0])-1)] for j in range(len(data_as_list))]
	get_class_labels = lambda data_as_list: [class_numerical_mapping[data_as_list[i][len(data_as_list[0])-1]] for i in range(len(data_as_list))]

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

	#print(float(count_correct))
	#print(len(predicted_class_labels))

	fitness = float(count_correct) / len(predicted_class_labels)

	return fitness

def count_1s(binary_string):
	return sum([int(i) for i in list(binary_string)])

if __name__ == '__main__':

	f = "musk/clean1.data"
	#f = "vehicle/vehicle.dat"

	NUMBER_OF_FEATURES = 167 if "musk" in f else 18 # this is for the musk data set
	class_numerical_mapping = {"non-musk":0, "musk":1} if "musk" in f else{"van":0, "saab":1, "bus":2, "opel":3}

	df = DataLoader.load_data(f)

	print(df)
	
	training_data, testing_data = train_test_split(df.T, test_size=0.2, random_state=random.randint(0,100))

	problem = ProblemWrapper(n_var=NUMBER_OF_FEATURES, n_obj=2, xl=0, xu=1, type_var=bool) # xl is lower bound, xu is upper bound

	algorithm = NSGA2(pop_size=5, sampling=get_sampling("bin_random"), crossover=get_crossover("bin_hux"), mutation=get_mutation("bin_bitflip"), eliminate_duplicates=True)

	stop_criteria = ('n_gen', 5)

	results = minimize(problem=problem, algorithm=algorithm, termination=stop_criteria)

	fitnesses = list()
	for result in results.X.astype(int):
		result = ''.join([str(i) for i in result])

		print(result)
		print(count_1s(result))

		fitnesses.append(wrapper_based_fitness_function(result))

	print(fitnesses)