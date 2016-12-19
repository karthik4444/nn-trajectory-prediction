import pickle
import sys
import numpy as np

def save_processed_scene(scene, s):
	pickle.dump(scene, open('train_data/pooling/scene' + str(s) + '/scene.pickle', 'wb'))

def load_processed_scene(s):
	return pickle.load(open('train_data/pooling/scene' + str(s) + '/scene.pickle', 'rb'))

def save_training_set(X, y, s, c):
	pickle.dump(X, open('train_data/naive/' + c + '/X' + str(s) + '.pickle', 'wb'))
	pickle.dump(y, open('train_data/naive/' + c + '/y' + str(s) + '.pickle', 'wb'))

def load_training_set(s, c):
	X = pickle.load(open('train_data/naive/' + c + '/X' + str(s) + '.pickle', 'rb'))
	y = pickle.load(open('train_data/naive/' + c + '/y' + str(s) + '.pickle', 'rb'))
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

def load_model(model, is_pooling_model, category):
	model_type = "pooling_model" if (is_pooling_model) else "naive_model"

	npzfile = np.load('models/' + category.lower() + '/' + model_type + '.npz')
	D, E, U, W, V, b, c = [npzfile["D"],
					 npzfile["E"], 
					 npzfile["U"], 
					 npzfile["W"],
					 npzfile["V"],
					 npzfile["b"],
					 npzfile["c"]]

	model.hidden_dim = E.shape[0]
	model.input_dim = E.shape[1]
	model.output_dim = c.shape[0]
	model.E.set_value(E)
	model.U.set_value(U)
	model.W.set_value(W)
	model.V.set_value(V)
	model.b.set_value(b)
	model.c.set_value(c)

	if is_pooling_model:
		model.D.set_value(D)
