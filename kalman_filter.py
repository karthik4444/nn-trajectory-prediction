import pickle
import numpy as np
import pdb
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

data_full = pickle.load(open('train_data/annotations_velocity.pickle'))
#print(data_full)

xMin = float("inf")
xMax = -float("inf")
yMin = float("inf")
yMax = -float("inf")

for value in data_full:
#    pdb.set_trace();
    for item in value:
        x = item[0]
        y = item[1]
        if x < xMin:
            xMin = x
        if x > xMax:
            xMax = x
        if y < yMin:
            yMin = y
        if y > yMax:
            yMax = y

#for i in xrange(len(data_full)):
#    row_len = len(data_full[i])
#    for j in xrange(row_len):
#        pdb.set_trace()
#        data_full[i][j][0] = (data_full[i][j][0] - xMin)/(xMax - xMin);
#        data_full[i][j][1] = (data_full[i][j][1] - yMin)/(yMax - yMin);
#print(data_full)

zeros = np.zeros((4, 1))
for i in xrange(1,2,1):
    kf = KalmanFilter(transition_matrices = np.identity(4),
            observation_matrices = np.identity(4),
            transition_offsets = np.zeros(4))
#    pdb.set_trace()
    value = data_full[i]
    print(len(value))
    measurements = value[0:5]
    kf = kf.em(measurements, n_iter = 5, em_vars = ['transition_matrices', 'transition_offsets'])  #, 'transition_covariance'])
    transition_matrix = kf.transition_matrices
    #print(transition_matrix)
    transition_offsets = kf.transition_offsets
    #transition_offsets = [ 0.64221677,  0.64221677,  0.37319751,  0.37319751]
    #transition_matrix = [[ 0.50294068,  0.47701169,  0.35751601,  0.35751601], [ 0.47701169,  0.50294068,  0.35751601,  0.35751601],
    #        [ 0.09135407,  0.09135407,  0.17646728,  0.1505383 ], [ 0.09135407,  0.09135407,  0.1505383 ,  0.17646728]]
    #transition_matrix = [[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 1, 0], [0, 0, 0, 1]]
    #transition_covariance = kf.transition_covariance
    #print(transition_covariance)
    print(value);
    #print(measurements)
    prev = value[4];
    print(measurements)
    print(prev);
    guesses = [];
#    print(prev);
#    print(transition_matrix);
#    print(transition_offsets);
    guesses.append(prev)
    #pdb.set_trace()
    for j in xrange(3):
        pred = np.dot(transition_matrix, prev) + transition_offsets #+ np.random.normal(zeros, transition_covariance)
        guesses.append(pred);
#        print(prev);
#        print(pred);
        prev = pred;

    x_pred_cord = [item[0] for item in guesses];
    y_pred_cord = [item[1] for item in guesses];

    x_cord = [item[0] for item in value];
    y_cord = [item[1] for item in value];

    plt.plot(x_pred_cord, y_pred_cord, '-o')
    plt.plot(x_cord, y_cord, '.r-')

    plt.show()


