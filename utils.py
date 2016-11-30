import pickle
import sys
import numpy as np

sys.setrecursionlimit(10000)

def save_processed_data(x, y):
	pickle.dump(x, open('train_data/X_train.pickle', 'wb'))
	pickle.dump(y, open('train_data/y_train.pickle', 'wb'))

def load_processed_data():
	return pickle.load(open('train_data/X_train.pickle', 'rb')), pickle.load(open('train_data/y_train.pickle', 'rb'))

def log_time_remaining(step_duration, num_examples, num_epochs, epoch):
	time_remaining = (step_duration * num_examples * (num_epochs - epoch))
	hours = int(time_remaining / 3600)
	minutes = int((time_remaining % 3600) / 60)
	print("TIME REMAINING: {} h {} min".format(hours, minutes))
def save_model(model):
	U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
	np.savez('models/baseline_gru', U=U, V=V, W=W)

def load_model(model):
	npzfile = np.load('models/baseline_gru')
	U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
	model.hidden_dim = U.shape[1]
	model.input_dim = U.shape[2]
	model.output_dim = V.shape[0]
	model.U.set_value(U)
	model.V.set_value(V)
	model.W.set_value(W)