import numpy as np
import theano
from utils import *

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
	data_seq_eth[val[1]].append((val[2], val[4]))

#eliminating short trajectories (threshold = minimum trajectory length of 9)
keys = dict(data_seq_eth).keys()
for i in keys:
	if (len(data_seq_eth[i]) < 9):
		del data_seq_eth[i]

#Each training example: first eight time steps is x and remaining time steps is y
X_train = [value[:4] for value in data_seq_eth.values()]
y_train = [value[4:] for value in data_seq_eth.values()]

save_processed_data(X_train, 'processed_data/X_train.pickle')
save_processed_data(y_train, 'processed_data/y_train.pickle')

'''	
X_train = np.asarray([i[:-1] for i in data_seq_eth.values()])
y_train = np.asarray([i[1:] for i in data_seq_eth.values()])


with open('processed_data/X_train.pickle', 'wb') as f:
	pickle.dump(X_train, f)

with open('processed_data/y_train.pickle', 'wb') as f:
	pickle.dump(y_train, f)

'''
