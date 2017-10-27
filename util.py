import numpy as np


def shuffle_data(X,Y):
    permutation = np.random.permutation(len(X))
    return (X[permutation],Y[permutation])

def load_data():
    # load data
	dataX = np.load("data/dataX.npy")   # images
	dataY = np.load("data/dataY.npy")  # labels
	dataX = dataX[:8000]
	dataY = dataY[:8000]

	# formated labels to one-hot format
	tmp = np.zeros([len(dataY), 10])
	for i in range(len(dataY)):
		tmp[i][int(dataY[i])] = 1

	dataY = tmp
	dataX, dataY = shuffle_data(dataX, dataY)
	train_index = int(len(dataX)*0.8)
	train_data = (dataX[:train_index],dataY[:train_index])
	test_data = (dataX[train_index:],dataY[train_index:])
	return (train_data,test_data)


    