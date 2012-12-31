# Chapter 5: Logistic Regression
from numpy import *
import random

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
    """this gradient ascent function returns the weights as
    a numpy matrix"""
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
    function result if it is a matrix.
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

def stoc_grad_ascent0(datamatrix, class_labels):
    """this gradient ascent function returns the weights
    as a numpy array"""
    m, n = shape(datamatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(datamatrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * datamatrix[i]
    return weights

def stoc_grad_ascent1(datamatrix, class_labels, num_iter=150):
    """this gradient ascent function returns the weights
    as a numpy array"""
    m, n = shape(datamatrix)
    weights = ones(n)

    for j in range(num_iter):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(datamatrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * datamatrix[rand_index]
            del(data_index[rand_index])
    return weights

def classify_vector(in_x, weights):
    return 1.0 if sigmoid(sum(in_x * weights)) > 0.5 else 0.0

def colic_test():
    training_set = []
    training_labels = []

    with open('horseColicTraining.txt') as train_file:
        for line in train_file.readlines():
            row = line.strip().split('\t')
            training_set.append([float(row[i]) for i in range(21)])
            training_labels.append(float(row[21]))

    train_weights = stoc_grad_ascent1(array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0

    with open('horseColicTest.txt') as test_file:
        for line in test_file.readlines():
            num_test_vec += 1.0
            row = line.strip().split('\t')
            line_arr = [float(row[i]) for i in range(21)]
            if int(classify_vector(array(line_arr), train_weights)) != int(row[21]):
                error_count += 1
    
    error_rate = float(error_count) / num_test_vec
    print 'the error rate of this test is: %f' % error_rate
    return error_rate

def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print 'after %d iterations the average error is: %f' % (num_tests, error_sum / float(num_tests))
            
