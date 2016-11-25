#training script for neural nets
import numpy as np
from simple_gru import SimpleGRU
import time
from utils import *

#loading data
X_train, y_train = load_processed_data()  

#hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 2
__HIDDEN_DIM = 128
__NUM_EPOCHS = 100
__LEARNING_RATE = 0.003

#defining training iterations
def train(model, x_train, y_train, learning_rate, num_epochs, step_duration,  evaluate_loss_after=5):
	num_training_examples = len(y_train)
	for epoch in range(num_epochs):
		time_remaining = (step_duration * num_training_examples) * (num_epochs - epoch)
		hours = time_remaining / 3600
		minutes = (time_remaining % 3600) / 60
		print("EPOCH: %d /100", epoch+1)
		print("TIME REMAINING: %d h %d min", hours, minutes)
		#console updates on training progress
		if (epoch % evaluate_loss_after == 0):
			print("CURRENT LOSS IS %f", model.loss())
		#training
		for example in range(num_training_examples):
			model.sgd_step(X_train[example], y_train[example], learning_rate)

#test performance with one gradient descent step
testModel = SimpleGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], __LEARNING_RATE)
t2 = time.time()
step_duration = t2 - t1

#train model
model = SimpleGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
train(model, X_train, y_train, __LEARNING_RATE, __NUM_EPOCHS, step_duration)