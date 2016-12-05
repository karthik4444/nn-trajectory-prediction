from utils import *

#processing data for seq_eth dataset
dataset_seq_eth = open('annotations.txt')

#data_seq_eth is a dictionary structued as {id: trajectory} i.e. {47: (1,2), (1,3) ...}
data_seq_eth = {}

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
        if not val[0] in data_seq_eth:
            data_seq_eth[val[0]] = []
	data_seq_eth[val[0]].append((x, y))

annotations_raw = [value for value in data_seq_eth.values()]

pickle.dump(annotations_raw, open('train_data/annotations_raw.pickle', 'wb'))
