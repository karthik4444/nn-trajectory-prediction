import pickle
import sys
import numpy as np

def save_processed_scene(scene):
	pickle.dump(scene, open('train_data/scene.pickle', 'wb'))

def load_processed_scene():
	return pickle.load(open('train_data/scene.pickle', 'rb'))

def save_training_set(X, y):
	pickle.dump(X, open('train_data/X.pickle', 'wb'))
	pickle.dump(y, open('train_data/y.pickle', 'wb'))

def load_training_set():
	X = pickle.load(open('train_data/X.pickle', 'rb'))
	y = pickle.load(open('train_data/y.pickle', 'rb'))
	return X,y

def log_time_remaining(step_duration, num_examples, num_epochs, epoch):
	time_remaining = (step_duration * num_examples * (num_epochs - epoch))
	hours = int(time_remaining / 3600)
	minutes = int((time_remaining % 3600) / 60)
	print("TIME REMAINING: {} h {} min".format(hours, minutes))

def save_model(model, category, is_pooling_model):
	E, U, W, V, b, c = [model.E.get_value(), 
					model.U.get_value(), 
					model.W.get_value(), 
					model.V.get_value(),
					model.b.get_value(),
					model.c.get_value()]
	
	model_type = "naive_model"
	D = np.empty([1])

	if is_pooling_model:
		D = model.D.get_value()
		model_type = "pooling_model"

	np.savez('models/' + category.lower() + '/' + model_type, D=D, E=E, U=U, W=W, V=V, b=b, c=c)

def load_model(model):
	npzfile = np.load('models/biker/naive_model')
	U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
	model.hidden_dim = U.shape[1]
	model.input_dim = U.shape[2]
	model.output_dim = V.shape[0]
	model.U.set_value(U)
	model.V.set_value(V)
	model.W.set_value(W)