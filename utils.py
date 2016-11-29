import pickle

def save_processed_data(var, filename):
	f = open(filename, 'wb')
	pickle.dump(var, f)

def load_processed_data():
	return pickle.load(open('processed_data/X_train')), pickle.load(open('processed_data/y_train'))