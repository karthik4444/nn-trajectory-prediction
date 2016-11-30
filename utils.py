import pickle

def save_processed_data(var, filename):
	f = open(filename, 'wb')
	pickle.dump(var, f)

def load_processed_data():
	return pickle.load(open('processed_data/X_train.pickle', 'rb')), pickle.load(open('processed_data/y_train.pickle', 'rb'))

def log_time_remaining(step_duration, num_examples, num_epochs, epoch):
	time_remaining = (step_duration * num_examples * (num_epochs - epoch))
	hours = time_remaining / 3600
	minutes = (time_remaining % 3600) / 60
	print("EPOCH: {} /{}}".format(epoch+1, num_epochs))
	print("TIME REMAINING: {} h {} min".format(hours, minutes))
def save_model(model):
	f = open('model/baseline_gru.pickle', 'wb')
	pickle.dump(model, f)

def load_model():
	f = open('model/baseline_gru.pickle', 'rb')
	model = pickle.load
	return model