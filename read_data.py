from utils import *

#processing data for seq_eth dataset
dataset_seq_eth = open('annotations.txt')

#data_seq_eth is a dictionary structued as {id: trajectory} i.e. {47: (1,2), (1,3) ...}
data_seq_eth = {k+1: [] for k in range(1000000)}


while True:
	line = dataset_seq_eth.readline()
	if line == '':
		break
	row = line.split(" ")
	if (row[-1] != "\"Pedestrian\"\n"):
		continue
	val = [float(i) for i in row[:-1]]
	if((val[4] % 20) != 0):
		continue
	x = (val[1] + val[3])/2
	y = (val[2] + val[4])/2
	data_seq_eth[val[0]].append((x, y))




#eliminating short trajectories (threshold = minimum trajectory length of 9)
keys = dict(data_seq_eth).keys()
for i in keys:
	if (len(data_seq_eth[i]) < 9):
		del data_seq_eth[i]

#Each training example: first eight time steps is x and remaining time steps is y
X_train = [value[:8] for value in data_seq_eth.values()]
y_train = [value[8:] for value in data_seq_eth.values()]

save_processed_data(X_train, y_train)
