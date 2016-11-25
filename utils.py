import pickle

def save_processed_data(var, filename):
	f = open(filename, 'wb')
	pickle.dump(var, f)