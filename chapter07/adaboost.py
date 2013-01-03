## Chapter 7 - Ada Boost example
from numpy import *

def load_data_simple():
    dat_mat = matrix([[1.,  2.1],
                      [2.,  1.1],
                      [1.3, 1. ],
                      [1.,  1. ],
                      [2.,  1. ]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels

