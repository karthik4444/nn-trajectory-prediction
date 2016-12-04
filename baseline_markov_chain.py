import pickle
import numpy as np
import pdb
from sklearn.metrics import mean_squared_error

X_full = pickle.load(open('processed_data/X_train.pickle'))
y_full = pickle.load(open('processed_data/y_train.pickle'))

#calculate mean and variance over samples

#take data and make them a series of data points

num_samples = X_full.shape[0]
num_train = int(0.8 * num_samples)
X_train = X_full[0: num_train]
X_test = X_full[num_train:]
Y_test = y_full[num_train:]
Y_train = y_full[0: num_train]
delta = []

# Dimensions:
#('minX = ', -6.9563657)
#('maxX = ', 13.868879)
#('minY = ', -2.6931833)
#('maxY = ', 13.287946)
x_dim = 11
y_dim = 9
total_classes = x_dim * y_dim
transition = np.zeros((total_classes + 1, total_classes + 1))

def getClass(x, y):
    return int((x + 7)/2) * y_dim + int((y + 3)/2)

#train
#1. calculate bivariate gaussian for step size.
#2. define transition matrix
for i in xrange(num_train):
    length = len(X_train[i]) - 1
    for j in xrange(length):
        x1, y1 = X_train[i][j]
        x2, y2 = X_train[i][j + 1]
        delta.append((x2 - x1, y2 - y1))
        x1Class = getClass(x1, y1)
        x2Class = getClass(x2, y2)
        try:
            transition[x1Class, x2Class] = transition[x1Class, x2Class] + 1
        except IndexError:
            print("x1Class ", x1Class)
            print("x2Class ", x2Class)
            print("(x1, y1) = ", x1, y1)
            print("(x2, y2) = ", x2, y2)
            print("(i, j) = ", i, j)

mean = np.mean(delta, axis = 0)
cov = np.cov(delta, rowvar = 0)

#for each item, find highest destiny
#continue generating random distribution until it falls into the density

num_test = X_test.shape[0]
pred = []
#predict/test
for i in xrange(num_test):
    length = len(X_test[i])
    pred.append([])
    for j in xrange(length):
        x1, y1 = X_test[i][j]
        x1Class = getClass(x1, y1)
        currHigh = (-1, -1)
        highVal = -1
        for _ in range(5):
            sample = np.random.multivariate_normal(mean, cov)
            x2, y2 = x1 + sample[0], x2 + sample[0]
            classVal = getClass(x2, y2)
            if (classVal > highVal):
                highVal = classVal
                currHigh = (x2, y2)
        pred[i].append(currHigh)

error = 0
nSamples = 0
for i in xrange(num_test):
    if (len(pred[i]) != 0):
        nSamples += len(pred[i])
        error += mean_squared_error(pred[i], Y_test[i])

print("Total Test Error: ", error)
print("Average Test Error: ", error/nSamples)

#predict/train
pred = []
for i in xrange(num_train):
    length = len(X_train[i])
    pred.append([])
    for j in xrange(length):
        x1, y1 = X_train[i][j]
        x1Class = getClass(x1, y1)
        currHigh = (-1, -1)
        highVal = -1
	for _ in range(5):
            sample = np.random.multivariate_normal(mean, cov)
            x2, y2 = x1 + sample[0], x2 + sample[0]
            classVal = getClass(x2, y2)
            if (classVal > highVal):
                highVal = classVal
                currHigh = (x2, y2)
        pred[i].append(currHigh)

error = 0
nSamples = 0
for i in xrange(num_train):
    #pdb.set_trace()
    if (len(pred[i]) != 0):
        nSamples += len(pred[i])
        error += mean_squared_error(pred[i], Y_train[i])

print("Total Train Error: ", error)
print("Average Train Error: ", error/nSamples)




