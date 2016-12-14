
from utils import *

#processing data for seq_eth dataset
dataset = open('annotations/deathCircle/video0/annotations.txt')
scene = {}

while True:
	line = dataset.readline()
	if line == '':
		break
	row = line.split(" ")
	frame = int(row[5])
	if frame % 18 != 0:
		continue

	x = (int(row[1]) + int(row[3])) / 2
	y = (int(row[2]) + int(row[4])) / 2
	label = row[-1][1:-2]
	member_id = int(row[0])
	info = [member_id, (x,y), label]
	if frame in scene:
		scene[frame].append(info)
	else:
		scene[frame] = [info]


save_processed_scene(scene)

def extract_naive_dataset(s):
	frames = s.keys()
	frames = sorted(frames)
	occupants = {}
	for frame in frames:
		for member in s[frame]:
			if member[2] == "Biker":
				if member[0] not in occupants: occupants[member[0]] = []
				occupants[member[0]].append(member[1])
	x_train = [occupants[member][:50] for member in occupants if len(occupants[member]) > 50]
	y_train = [occupants[member][50:] for member in occupants if len(occupants[member]) > 50]
	return x_train, y_train

X, y = extract_naive_dataset(scene)
save_training_set(X,y)



