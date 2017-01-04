from utils import *

#reading and formatting data from dataset annotations

#training on first 4 scenes and testing on last (holdout).
NUM_SCENES = 5
TEST_SCENE = 4
classes = ["Pedestrian", "Biker", "Skater", "Cart"]

for s in range(NUM_SCENES):
	#load annotations
	dataset = open('annotations/deathCircle/video' + str(s) + '/annotations.txt')
	#dictionary to hold parsed details
	scene = {}

	while True:
		line = dataset.readline()
		if line == '':
			break
		row = line.split(" ")
		frame = int(row[5])
		if frame % 15 != 0:
			continue

		x = (int(row[1]) + int(row[3])) / 2
		y = (int(row[2]) + int(row[4])) / 2
		label = row[-1][1:-2]
		#skip sparse busses and resolve cars as carts
		if label == "Bus":
			continue
		if label == "Car":
			label = "Cart"
		member_id = int(row[0])
		info = [member_id, (x,y), label]
		if frame in scene:
			scene[frame].append(info)
		else:
			scene[frame] = [info]

	#spearate parsed info into the three dictionaries (reduces complexity while training)
	#outlay_dict contains position per frame. class_dict contains classification per member-id. path_dict contains path thus far per member-id
	outlay_dict, class_dict, path_dict = {}, {}, {}
	frames = scene.keys()
	frames = sorted(frames)
	for frame in frames:
		outlay_dict[frame], path_dict[frame] = {}, {}
		for obj in scene[frame]:
			outlay_dict[frame][obj[0]] = obj[1]
			class_dict[obj[0]] = obj[2]

			if frame == 0:
				path_dict[frame][obj[0]] = [obj[1]]
				continue

			prev_frame = frames[frames.index(frame) - 1]
			if obj[0] not in path_dict[prev_frame]:
				path_dict[frame][obj[0]] = [obj[1]]
			else:
				path_dict[frame][obj[0]] = path_dict[prev_frame][obj[0]] + [obj[1]]

	save_processed_scene([outlay_dict, class_dict, path_dict], s)

	#constructing a simpler dataset for naive training
	def extract_naive_dataset(s, c):
		frames = s.keys()
		frames = sorted(frames)
		occupants = {}
		for frame in frames:
			for member in s[frame]:
				if member[2] == c:
					if member[0] not in occupants: occupants[member[0]] = []
					occupants[member[0]].append(member[1])
		x_train = []
		y_train = []
		for member in occupants:
			traj = occupants[member]
			if len(traj) < 19:
				continue
			for i in range(0, len(traj) - 18):
				x_train.append(traj[i:i+18])
				y_train.append(traj[i+18])
		return x_train, y_train

	for c in classes:
		X, y = extract_naive_dataset(scene, c)
		save_training_set(X, y, s, c)



