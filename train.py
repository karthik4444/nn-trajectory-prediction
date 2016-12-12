#training script for neural nets
import numpy as np
from pooling_gru import PoolingGRU
from utils import *
import math
import pdb

_id = 0
_position = 1
_class = 2

#hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 2
__HIDDEN_DIM = 128
__NUM_EPOCHS = 10 #100
__LEARNING_RATE = 0.003
__POOLING_SIZE = 20

def add_to_tracking_list(k, v, d):
	if k in d:
		d[k].append(v)
	else:  
		d[k] = [v]

def map_tensor_index(pos, ref_pos):
	x = math.ceil((pos[0] - ref_pos[0])/8) + 9
	y = math.ceil((pos[1] - ref_pos[1])/8) + 9
	return (int(x),int(y))

def pool_hidden_states(obj_id, position, hidden_states):
	pooled_tensor = [[[0] * __HIDDEN_DIM] * __POOLING_SIZE] * __POOLING_SIZE
	bound = __POOLING_SIZE * 8 / 2 
	window_limits_upper_bound = (position[0] + bound, position[1] + bound)
	window_limits_lower_bound = (position[0] - bound, position[1] - bound)
	for ID in hidden_states:
		if ID != obj_id:
			pos = hidden_states[ID][0]
			within_upper_bound = (pos[0] <= window_limits_upper_bound[0]) & (pos[1] <= window_limits_upper_bound[1])
			within_lower_bound = (pos[0] > window_limits_lower_bound[0]) & (pos[1] > window_limits_lower_bound[1])
			if within_upper_bound & within_lower_bound:
				x,y = map_tensor_index(pos, position)
				pooled_tensor[x][y] = hidden_states[ID][1]
	return pooled_tensor

def step_through_scene(scene, models):
	print("STEPPING THROUGH SCENE")
	frames = scene.keys()
	frames = sorted(frames)
	position_tracker = {}
	neighbor_tracker = {}
	ground_truth_path = {}
	obj_class = {}
	for frame in frames:
		pdb.set_trace()
		#parse through scene to gather hidden states of objects
		print("frame {} / {}".format(frame, len(frames)))
		hidden_states = {}
		for obj in scene[frame]:
			obj_class[obj[_id]] = obj[_class]
			if (obj[_id] in ground_truth_path):
				h = models[obj[_class]].get_hidden(ground_truth_path[obj[_id]], neighbor_tracker[obj[_id]])
				h = h.tolist()
				hidden_states[obj[_id]] = (obj[_position], h)

		#parse through scene to gather training criteria
		for obj in scene[frame]:
			nsteps = len(ground_truth_path[obj[_id]]) if (obj[_id] in ground_truth_path) else 0
			pos = models[obj[_class]].predict(position_tracker[obj[_id]], neighbor_tracker[obj[_id]]).tolist() if (nsteps > 8) else obj[_position]
			
			add_to_tracking_list(obj[_id], obj[_position], ground_truth_path)
			add_to_tracking_list(obj[_id], pos, position_tracker)

			pooled_tensor = pool_hidden_states(obj[_id], pos, hidden_states)
			add_to_tracking_list(obj[_id], pooled_tensor, neighbor_tracker)
	return obj_class, neighbor_tracker, ground_truth_path

def train(classes, models, scene, learning_rates, check_cost_after, num_epochs): 
	print("TRAINING MODELS")
	frames = scene.keys()
	frames = sorted(frames)
	previous_costs = {model: math.inf for model in classes}
	for epoch in range(num_epochs):

		print("EPOCH {} / {}".format(epoch+1, num_epochs))

		obj_class, neighbor_tracker, ground_truth_path = step_through_scene(scene, models)
		
		if epoch % check_cost_after == 0:
			losses = {model: [] for model in classes}
			for _id in ground_truth_path:
				if len(ground_truth_path[_id]) > 8:
					c = obj_class[_id]
					losses[c].append(models[c].loss(ground_truth_path[_id][:8], neighbor_tracker[_id][:8], ground_truth_path[_id][8:], neighbor_tracker[_id][8:]))
			for c in classes:
				cost = sum(losses[c])/len(losses[c]) if len(losses[c]) else 0
				print ("{}: COST IS {}".format(c, cost))
				if cost > previous_costs[c]:
					learning_rates[c] *= 0.5
					print ("{} LEARNING RATE HAS BEEN HALVED".format(c))
				previous_costs[c] = cost

		for _id in ground_truth_path:
			if len(ground_truth_path[_id]) > 8:
				c = obj_class[_id]
				models[c].sgd_step(ground_truth_path[_id][:8], neighbor_tracker[_id][:8], ground_truth_path[_id][8:], neighbor_tracker[_id][8:], learning_rates[c])
					

print("CREATING MODELS")
classes = ["Pedestrian", "Biker", "Skater", "Cart", "Bus", "Car"]
models = {label: PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __POOLING_SIZE, __HIDDEN_DIM) for label in classes}

#loading data
scene = load_processed_scene()

learning_rates = {model : __LEARNING_RATE for model in classes}
check_cost_after = 5

train(classes, models, scene, learning_rates, check_cost_after, __NUM_EPOCHS)

				