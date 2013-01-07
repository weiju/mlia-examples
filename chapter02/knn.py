import numpy as np
import os
import operator

# **********************************************************************
# **** knn.py
# **** An adaptation of the k-nearest neighbors example from
# **** "Machine Learning in Action" by Peter Harrington
# **********************************************************************

def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(in_x, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_map = np.tile(in_x, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_map**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_dist_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def file2matrix(filename):
    with open(filename) as infile:
        all_lines = infile.readlines()

    num_lines = len(all_lines)
    return_matrix = np.zeros((num_lines, 3))
    index = 0
    rows = [line.strip().split('\t') for line in all_lines]

    # see errata: author switched from text classes to int classes
    class_labels = [int(row[-1]) for row in rows]

    for row in rows:
        return_matrix[index,:] = row[0:3]
        index += 1
    return return_matrix, class_labels

def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals

######################################################################
####
#### Dating site example
####
######################################################################

def dating_class_test():
    ho_ratio = 0.1
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i,:],
                                      norm_mat[num_test_vecs:m,:],
                                      dating_labels[num_test_vecs:m], 3)
        print ("the classifier came back with: %d, the real answer is: %d"
               % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print "the total error rate is: %f" % (error_count / float(num_test_vecs))

def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input('percentage of time spent playing video games ?'))
    ff_miles = float(raw_input('frequent flier miles earned per year ?'))
    ice_cream = float(raw_input('liters of ice cream consumed per year ?'))

    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges,
                                 norm_mat,
                                  dating_labels, 3)
    print "You will probably like this person: ", result_list[classifier_result - 1]

######################################################################
####
#### Handwriting Recognition example
####
######################################################################

def img2vector(filename):
    """read the data in 32x32 0/1 format"""
    return_vect = np.zeros((1, 1024))
    with open(filename) as infile:
        for i in range(32):
            row = infile.readline()
            for j in range(32):
                return_vect[0, 32*i+j] = int(row[j])
    return return_vect

def handwriting_class_test():
    def extract_classnum(filename):
        file_str = filename.split('.')[0]
        return int(file_str.split('_')[0])
        
    hw_labels = []
    training_files = os.listdir('trainingDigits')
    training_set_size = len(training_files)
    training_mat = np.zeros((training_set_size, 1024))
    for i in range(training_set_size):
        filename = training_files[i]
        class_num = extract_classnum(filename)
        hw_labels.append(class_num)
        training_mat[i,:] = img2vector('trainingDigits/%s' % filename)

    test_files = os.listdir('testDigits')
    error_count = 0.0
    test_set_size = len(test_files)
    for i in range(test_set_size):
        filename = test_files[i]
        class_num = extract_classnum(filename)
        vector_under_test = img2vector('testDigits/%s' % filename)
        classifier_result = classify0(vector_under_test, training_mat,
                                      hw_labels, 3)
        print ("the classifier came back with: %d, the real answer is: %d"
               % (classifier_result, class_num))
        if classifier_result != class_num:
            error_count += 1.0
    print "\nthe total number of errors is: %d" % error_count
    print "\nthe total error rate is: %f" % (error_count / float(test_set_size))
