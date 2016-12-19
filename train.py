#training script for neural nets
import numpy as np
from pooling_gru import PoolingGRU
from baseline.naive_gru import NaiveGRU
from utils import *
import pdb
import argparse
import math

_id = 0
_position = 1
_class = 2

#hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 2
__HIDDEN_DIM = 128
__NUM_EPOCHS = 100
__LEARNING_RATE = 0.003
__POOLING_SIZE = 20
__NUM_SCENES = 4

classes = ["Pedestrian", "Biker", "Skater", "Cart"]

def map_tensor_index(pos, ref_pos):
	x = math.ceil((pos[0] - ref_pos[0])/8) + 9
	y = math.ceil((pos[1] - ref_pos[1])/8) + 9
	return (int(x),int(y))

def pool_hidden_states(member_id, position, hidden_states):
	pooled_tensor = [[[0] * __HIDDEN_DIM] * __POOLING_SIZE] * __POOLING_SIZE
	bound = __POOLING_SIZE * 8 / 2 
	window_limits_upper_bound = (position[0] + bound, position[1] + bound)
	window_limits_lower_bound = (position[0] - bound, position[1] - bound)
	for ID in hidden_states:
		if ID != member_id:
			pos = hidden_states[ID][0]
			within_upper_bound = (pos[0] <= window_limits_upper_bound[0]) and (pos[1] <= window_limits_upper_bound[1])
			within_lower_bound = (pos[0] > window_limits_lower_bound[0]) and (pos[1] > window_limits_lower_bound[1])
			if within_upper_bound and within_lower_bound:
				x,y = map_tensor_index(pos, position)
				pooled_tensor[x][y] = hidden_states[ID][1]
	return pooled_tensor

def step_through_scene(models, scene, learning_rates, epoch, num_epochs, calculate_loss):
	outlay_dict = scene[0]
	class_dict = scene[1]
	path_dict = scene[2]
	frames = outlay_dict.keys()
	frames = sorted(frames)
	cost = {c: [] for c in classes}
	prev_hidden_states = {}
	pooled_tensors = {}
	for frame in frames:
			print "EPOCH {} / {} : FRAME {} / {}".format(epoch+1, num_epochs, frame, frames[-1])
			frame_occupants = outlay_dict[frame].keys()
			hidden_states = {}
			for occupant in frame_occupants:
				if occupant not in pooled_tensors:
					pooled_tensors[occupant] = []
				#pool tensors
				position = outlay_dict[frame][occupant]
				c = class_dict[occupant]
				pooled_tensor = pool_hidden_states(occupant, position, hidden_states)
				pooled_tensors[occupant].append(pooled_tensor)
				h = prev_hidden_states[occupant][1] if occupant in prev_hidden_states else [0] * __HIDDEN_DIM
				ns, nh = models[c].time_step(position, pooled_tensor, h)
				hidden_states[occupant] = (position, nh.tolist())

				path = path_dict[frame][occupant]
				if len(path) > 18:
					y = path[-1]
					x = path[-19:-1]
					H = pooled_tensors[occupant][-18:]
					if calculate_loss:
						cost[c].append(models[c].loss(x, H, y))
					else:
						models[c].sgd_step(x, H, y, learning_rates[c])
			prev_hidden_states = hidden_states
	if calculate_loss:
		return {c: sum(cost[c])/len(cost[c]) for c in cost}


def train_with_pooling(models, num_scenes, learning_rates, num_epochs, evaluate_loss_after=5):
	prev_cost = {c: float("inf") for c in classes}
	for epoch in range(num_epochs):
		cost = {c: 0 for c in classes}
		for s in range(num_scenes):
			scene = load_processed_scene(s)
			if (epoch + 1) % evaluate_loss_after == 0:
				cost_update = step_through_scene(models, scene, learning_rates, epoch, num_epochs, True)
				cost = {c : cost[c] + cost_update[c] for c in cost}

				if (s+1) == num_scenes:
					for c in cost:
						print "{} COST : {}".format(c, cost[c])
						if cost[c] > prev_cost[c]:
							learning_rates[c] *= 0.5
							print "LEARNING RATE FOR {} WAS HALVED".format(c)
					prev_cost = cost

			step_through_scene(models, scene, learning_rates, epoch, num_epochs, False)

			for c in models:
				save_model(models[c], c, True)


def train_naively(model, num_scenes, learning_rate, num_epochs, category, evaluate_loss_after=5):
	last_cost = float("inf")
	for epoch in range(num_epochs):
		print "EPOCH: {} /{}".format(epoch+1, num_epochs)
		cost = 0
		for s in range(num_scenes):
			x_train, y_train = load_training_set(s, category)
			print "SCENE: {} /{}".format(s+1, num_scenes)
			if ((epoch+1) % evaluate_loss_after == 0):
				cost += model.cost(x_train, y_train)
				if (s+1) == num_scenes:
					print("CURRENT COST IS {}".format(cost))
					if (cost > last_cost):
						learning_rate = learning_rate * 0.5
						print "Learning rate was halved to {}".format(learning_rate) 
					last_cost = cost

			for example in range(len(y_train)):
				model.sgd_step(x_train[example], y_train[example], learning_rate)
			save_model(model, category, False)


parser  = argparse.ArgumentParser(description='Pick Training Mode.')
parser.add_argument('mode', type=str, nargs=1, help="which mode to use for training? either 'pooling' or 'naive'")
mode = parser.parse_args().mode[-1]

if mode == "pooling":
	print 'creating models' 
	models = {label: PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __POOLING_SIZE, __HIDDEN_DIM) for label in classes}
	learning_rates = {model : __LEARNING_RATE for model in classes}
	train_with_pooling(models, __NUM_SCENES, learning_rates, __NUM_EPOCHS)

elif mode == "naive":
	print 'creating model'
	model = NaiveGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
	CLASS = "Biker"
	train_naively(model, __NUM_SCENES, __LEARNING_RATE, __NUM_EPOCHS, CLASS)

else:
	print("enter a valid mode: either 'pooling' or 'naive'")





