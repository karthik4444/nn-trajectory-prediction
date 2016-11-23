#training script for neural nets
import numpy as np
from simple_gru import SimpleGRU
import time

#loading data
X_train = 

#hyperparameters
__INPUT_DIM = 2
__OUTPUT_DIM = 2
__HIDDEN_DIM = 128
__NUM_EPOCHS = 100
__LEARNING_RATE = 0.003

#defining training iterations
#NOT YET IMPLEMENTED

#training script
model = SimpleGRU(__INPUT_DIM, __OUTPUT_DIM, __HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], __LEARNING_RATE)
t2 = time.time()