
from utils import *

#processing data for seq_eth dataset
dataset_seq_eth = open('../raw_data/annotations.txt')
scene = {}

classes = {
	"Pedestrian": [1,0,0,0,0,0],
	"Biker": [0,1,0,0,0,0],
	"Skater": [0,0,1,0,0,0],
	"Cart": [0,0,0,1,0,0],
	"Bus": [0,0,0,0,1,0],
	"Car": [0,0,0,0,0,1]
}

while True:
	line = dataset_seq_eth.readline()
	if line == '':
		break
	row = line.split(" ")
	frame = int(row[5])
	x = (int(row[1]) + int(row[3])) / 2
	y = (int(row[2]) + int(row[4])) / 2
	label = row[-1][1:-2]
	info = [int(row[1]), (x,y), label,classes[label]]
	if frame in scene:
		scene[frame].append(info)
	else:
		scene[frame] = [info]

save_processed_scene(scene)

