#training script for neural nets
import numpy as np
from pooling_gru import PoolingGRU
import time
from utils import *

#loading data
scene = load_processed_scene() 

#hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 2
__HIDDEN_DIM = 128
__NUM_EPOCHS = 200
__LEARNING_RATE = 0.003

models = {
	"Pedestrian": PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM),
	"Biker": PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM),
	"Skater": PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM),
	"Cart": PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM),
	"Bus": PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM),
	"Car": PoolingGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
}

def pool_hidden_states(hidden_states, other_positions, position, id):
	pooling_bounds = []
	for i in range(32):
		for j in range(32):
			top_right_bound = () if () else ()
			bottom_left_bound = () if () else ()

#training iterations
def train():
	losses = []
	for epoch in range(num_epochs):
		scene_hidden_states = {}
		obj_history = {}
		for frame in scene.keys():
			layout = scene[frame]

			obj_in_scene = [info[0] for object in layout]
			id_holder = dict(current_scene_hidden_states).keys()
			for obj_id in keys_holder
				if obj_id not in obj_in_scene:
					#perform SGD and iterate model if enough steps
					del current_scene_hidden_states[obj_id]
			for obj_id in obj_in_scene:
				if obj_id not in past_scene_hidden_states:
					past_scene_hidden_states[obj_id] = [0] * hidden_size
			
			obj_positions = {obj[0] : obj[1] for obj in layout}
			for obj in layout:
				new_scene_hidden_states = {}
				obj_id = obj[0]
				obj_position = obj[1]
				obj_model = models[obj[2]]
				obj_encoded_position = obj[3]

				pooled_tensor = pool_hidden_states(scene_hidden_states ,obj_positions, obj_position, obj_id)


