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
__NUM_EPOCHS = 100 #100
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

def pool_hidden_states(member_id, position, hidden_states):
	pooled_tensor = [[[0] * __HIDDEN_DIM] * __POOLING_SIZE] * __POOLING_SIZE
	bound = __POOLING_SIZE * 8 / 2 
	window_limits_upper_bound = (position[0] + bound, position[1] + bound)
	window_limits_lower_bound = (position[0] - bound, position[1] - bound)
	for ID in hidden_states:
		if ID != member_id:
			pos = hidden_states[ID][0]
			within_upper_bound = (pos[0] <= window_limits_upper_bound[0]) & (pos[1] <= window_limits_upper_bound[1])
			within_lower_bound = (pos[0] > window_limits_lower_bound[0]) & (pos[1] > window_limits_lower_bound[1])
			if within_upper_bound & within_lower_bound:
				x,y = map_tensor_index(pos, position)
				pooled_tensor[x][y] = hidden_states[ID][1]
	return pooled_tensor

def step_through_scene(scene, models, epoch, num_epochs):
	print "STEPPING THROUGH SCENE" 
	frames = scene.keys()
	frames = sorted(frames)
	neighbor_tracker = {}
	ground_truth_path = {}
	member_class = {}
	for frame in frames:
		#parse through scene to gather hidden states of memberects
		print "EPOCH {} / {} : FRAME {} / {}".format(epoch, num_epochs, frame, frames[-1])
		hidden_states = {}
		for member in scene[frame]:
			member_class[member[_id]] = member[_class]
			if (member[_id] in ground_truth_path):
				h = models[member[_class]].get_hidden(ground_truth_path[member[_id]], neighbor_tracker[member[_id]])
				h = h.tolist()
				hidden_states[member[_id]] = (member[_position], h)

		#parse through scene to gather training criteria
		for member in scene[frame]:
			add_to_tracking_list(member[_id], member[_position], ground_truth_path)

			pooled_tensor = pool_hidden_states(member[_id], member[_position], hidden_states)
			add_to_tracking_list(member[_id], pooled_tensor, neighbor_tracker)

	return member_class, neighbor_tracker, ground_truth_path

def train_with_pooling(classes, models, scene, learning_rates, num_epochs, evaluate_loss_after=5): 
	print "TRAINING MODELS"
	frames = scene.keys()
	frames = sorted(frames)
	previous_costs = {model: float("inf") for model in classes}

	for epoch in range(num_epochs):
		member_class, neighbor_tracker, ground_truth_path = step_through_scene(scene, models, epoch+1, num_epochs)
		
		x_train, xH_train, y_train = {}, {}, {}
		for member in ground_truth_path:
			x_train[member], xH_train[member], y_train[member] = [], [], []
			c = member_class[member]
			for i in range(0, len(ground_truth_path[member])-30, 10):
				x_train[member].append(ground_truth_path[i:i+30])
				xH_train[member].append(neighbor_tracker[i:i+30])
				y_train[member].append(ground_truth_path[i+30])

		for c in classes:
			if ((epoch+1) % evaluate_loss_after == 0):
				cost = models[c].cost(x_train[c], xH_train[c], y_train[c])
				print "CURRENT COST IS {}".format(cost)

				if (cost > previous_costs[c]):
					learning_rates[c] *= 0.5
					print "{} LEARNING RATE HAS BEEN HALVED".format(c)
				previous_costs[c] = cost

			#training
			num_examples = len(y_train[c])
			for example in range(num_examples):
				models[c].sgd_step(x_train[c][example], xH_train[x][example], y_train[c][example], learning_rates[c])
			save_model(models[c], c, True)


		if epoch % evaluate_loss_after == 0:
			losses = {model: [] for model in classes}
			for _id in ground_truth_path:
				if len(ground_truth_path[_id]) > 30:
					c = member_class[_id]
					losses[c].append(models[c].loss(ground_truth_path[_id][:50], neighbor_tracker[_id][:50], ground_truth_path[_id][50:], neighbor_tracker[_id][50:]))
			for c in classes:
				cost = sum(losses[c])/len(losses[c]) if len(losses[c]) else 0
				print "{}: COST IS {}".format(c, cost)
				if cost > previous_costs[c]:
					learning_rates[c] *= 0.5
					print "{} LEARNING RATE HAS BEEN HALVED".format(c)
				previous_costs[c] = cost

		for _id in ground_truth_path:
			if len(ground_truth_path[_id]) > 50:
				c = member_class[_id]
				models[c].sgd_step(ground_truth_path[_id][:50], neighbor_tracker[_id][:50], ground_truth_path[_id][50:], neighbor_tracker[_id][50:], learning_rates[c])
		for c in models:
			save_model(models[c], c, True)

def train_naively(model, x_train, y_train, learning_rate, num_epochs, category, evaluate_loss_after=5):
	last_cost = float("inf")
	for epoch in range(num_epochs):
		print "EPOCH: {} /{}".format(epoch+1, num_epochs)
		if ((epoch+1) % evaluate_loss_after == 0):
			cost = model.cost(x_train, y_train)
			print("CURRENT COST IS {}".format(cost))
			if (cost > last_cost):
				learning_rate = learning_rate * 0.5
				print "Learning rate was halved to {}".format(learning_rate) 
			last_cost = cost
		#training
		for example in range(len(y_train)):
			model.sgd_step(x_train[example], y_train[example], learning_rate)
		save_model(model, category, False)


parser  = argparse.ArgumentParser(description='Pick Training Mode.')
parser.add_argument('mode', type=str, nargs=1, help="which mode to use for training? either 'pooling' or 'naive'")
mode = parser.parse_args().mode[-1]

if mode == "pooling":
	print 'creating models' 
	scene = load_processed_scene()
	classes = ["Pedestrian", "Biker", "Skater", "Cart"]
	learning_rates = {model : __LEARNING_RATE for model in classes}
	models = {label: PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __POOLING_SIZE, __HIDDEN_DIM) for label in classes}
	train_with_pooling(classes, models, scene, learning_rates, __NUM_EPOCHS)

elif mode == "naive":
	print('creating model')
	model = NaiveGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
	x_train, y_train = load_training_set()
	print('training model')
	train_naively(model, x_train, y_train, __LEARNING_RATE, __NUM_EPOCHS, "Biker")

else:
	print("enter a valid mode: either 'pooling' or 'naive'")


