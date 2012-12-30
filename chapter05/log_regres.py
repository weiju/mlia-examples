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
