import pickle
import sys
import numpy as np

def save_processed_scene(scene):
	pickle.dump(scene, open('train_data/scene.pickle', 'wb'))

def load_processed_scene():
	return pickle.load(open('train_data/scene.pickle', 'rb'))

def log_time_remaining(step_duration, num_examples, num_epochs, epoch):
	time_remaining = (step_duration * num_examples * (num_epochs - epoch))
	hours = int(time_remaining / 3600)
	minutes = int((time_remaining % 3600) / 60)
	print("TIME REMAINING: {} h {} min".format(hours, minutes))

def save_model(model):
	U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
	np.savez('model/baseline_gru', U=U, V=V, W=W)

def load_model(model):
	npzfile = np.load('model/baseline_gru')
	U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
	model.hidden_dim = U.shape[1]
	model.input_dim = U.shape[2]
	model.output_dim = V.shape[0]
	model.U.set_value(U)
	model.V.set_value(V)
	model.W.set_value(W)