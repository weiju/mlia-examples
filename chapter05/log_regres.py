# Chapter 5: Logistic Regression
from numpy import *

def load_dataset():
    data_mat = []
    label_mat = []
    with open('testSet.txt') as infile:
        for line in infile.readlines():
            row = line.strip().split()
            data_mat.append([1.0, float(row[0]), float(row[1])])
            label_mat.append(int(row[2]))
        return data_mat, label_mat

def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))

def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights

def plot_best_fit(weights):
    """Note that the book has an error here (see Errata)
    the weights parameter should be a numpy array, which
    can be obtained by calling getA() on the grad_ascent()
    function result
    """
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_dataset()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    xcoord1 = []
    ycoord1 = []
    xcoord2 = []
    ycoord2 = []

    for i in range(n):
        if int(label_mat[i]) == 1:
            xcoord1.append(data_arr[i, 1])
            ycoord1.append(data_arr[i, 2])
        else:
            xcoord2.append(data_arr[i, 1])
            ycoord2.append(data_arr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoord1, ycoord1, s=30, c='red', marker='s')
    ax.scatter(xcoord2, ycoord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
