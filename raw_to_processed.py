import numpy as np
import pickle

#processing data for seq_eth dataset
dataset_seq_eth = open('raw_data/ETH/seq_eth/obsmat.txt')

#there are 367 people in the scene
#data_seq_eth is a dictionary structued as {id: trajectory} i.e. {47: (1,2), (1,3) ...}
data_seq_eth = {k+1: [] for k in range(367)}

while True:
	line = dataset_seq_eth.readline()
	if line == '':
		break
	val = [float(i) for i in line.split("  ") if i != '']
	data_seq_eth[val[1]].append((val[2],val[4]))

X_train = np.asarray([i[:-1] for i in data_seq_eth.values()])
y_train = np.asarray([i[1:] for i in data_seq_eth.values()])


with open('processed_data/X_train.pickle', 'wb') as f:
	pickle.dump(X_train, f)

with open('processed_data/y_train.pickle', 'wb') as f:
	pickle.dump(y_train, f)

'''
Access the pickled variable with

with open('file_path', 'rb') as f:
    X_train = pickle.load(f)
'''