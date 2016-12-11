
from utils import *

#processing data for seq_eth dataset
dataset_seq_eth = open('annotations/bookstore/video0/annotations.txt')
scene = {}
count = 0

while True:
	line = dataset_seq_eth.readline()
	if line == '':
		break
	if count % 20 != 0:
		continue
	row = line.split(" ")
	frame = int(row[5])
	x = (int(row[1]) + int(row[3])) / 2
	y = (int(row[2]) + int(row[4])) / 2
	label = row[-1][1:-2]
	object_id = int(row[1])
	info = [object_id, (x,y), label]
	if frame in scene:
		scene[frame].append(info)
	else:
		scene[frame] = [info]
	count += 1

save_processed_scene(scene)

