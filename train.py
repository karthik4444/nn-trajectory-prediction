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

#training iterations
def train(model, x_train, y_train, learning_rate, num_epochs, step_duration,  evaluate_loss_after=1):
	for epoch in range(num_epochs):
		log_time_remaining(step_duration, len(y_train), num_epochs, epoch)
		if (epoch % evaluate_loss_after == 0):
			print("CURRENT COST IS {}".format(model.cost(x_train, y_train)))
		#training
		for example in range(len(y_train)):
			model.sgd_step(X_train[example], y_train[example], len(y_train[example]), learning_rate)

#test performance with one gradient descent step
testModel = SimpleGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
t1 = time.time()
testModel.sgd_step(X_train[10], y_train[10], len(y_train[10]), __LEARNING_RATE)
t2 = time.time()
step_duration = t2 - t1
print("one sgd step takes {} microseconds".format(step_duration * 1000))

model = SimpleGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
train(model, X_train, y_train, __LEARNING_RATE, __NUM_EPOCHS, step_duration)
