import pandas as pd

class DataLoader:

	@staticmethod
	def load_data(filename):
		
		# this will automatically close the file when we exit the block
		with open("data/musk/" + str(filename), "r") as f:
			lines = [line.split() for line in f.readlines()]
			str_list = [list(line[0].split(',')) for line in lines]
			
			data_list = list()
			for l in str_list:
				data_entry = list()
				data_class = "non-musk" if "non-musk" in l[0].lower() else "musk"
				conformation_name = l[1]
				for i, feature in enumerate(l):
					if i == 0 or i == 1 or i == (len(l)-1):
						continue
					data_entry.append(int(feature))
			
				#data_entry.append(conformation_name)
				data_entry.append(data_class)
				data_list.append(data_entry)
			
			data_dict = dict()
			for index, line in enumerate(data_list):
				data_dict[index] = line

			return pd.DataFrame(data_dict)