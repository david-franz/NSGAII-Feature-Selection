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
# this is directly copied from an old project- need to make work with this project
def wrapper_based_fitness_function(binary_string):
	if binary_string == ('0' * len(binary_string)):
		return 0

	# process this as a list here to ensure we remove the class label
	training_data_filtered, class_labels = filter_rows_with_binary_string(training_data, binary_string, True)
	df_as_list = convert_df_to_list(training_data_filtered)

	knn = KNeighborsClassifier(n_neighbors=10)

	knn.fit(df_as_list, class_labels)

	testing_data_filtered, testing_data_class_labels = filter_rows_with_binary_string(testing_data, binary_string, True)
	testing_data_filtered_as_list = convert_df_to_list(testing_data_filtered)

	predicted_class_labels = knn.predict(testing_data_filtered_as_list)

	count_correct = 0
	for i in range(len(predicted_class_labels)):
		if predicted_class_labels[i] == testing_data_class_labels[i]:
			count_correct += 1

	return float(count_correct) / len(predicted_class_labels)