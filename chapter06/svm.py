## Chapter 6 - Support Vector Machines
from numpy import *
import random


def load_dataset(filename):
    data_mat = []
    label_mat = []
    with open(filename) as infile:
        for line in infile.readlines():
            row = line.strip().split('\t')
            data_mat.append([float(row[0]), float(row[1])])
            label_mat.append(float(row[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, high, low):
    if aj > high:
        aj = high
    if low > aj:
        aj = low
    return aj


def smo_simple(datamat_in, class_labels, c, toler, max_iter):
    datamatrix = mat(datamat_in)
    label_mat = mat(class_labels).transpose()
    b = 0
    m, n = shape(datamatrix)
    alphas = mat(zeros((m, 1)))
    iteration = 0
    while iteration < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            f_xi = float(multiply(alphas, label_mat).T * \
                             (datamatrix * datamatrix[i,:].T)) + b
            e_i = f_xi - float(label_mat[i])
            if ((label_mat[i] * e_i < -toler) and (alphas[i] < c)) or \
                    ((label_mat[i] * e_i > toler) and (alphas[i] > 0)):
                j = select_j_rand(i, m)
                f_xj = float(multiply(alphas, label_mat).T * \
                                 (datamatrix * datamatrix[j,:].T)) + b
                e_j = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                if label_mat[i] != label_mat[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])

                if low == high:
                    print 'low == high'
                    continue

                eta = 2.0 * datamatrix[i,:] * datamatrix[j,:].T - \
                    datamatrix[i,:] * datamatrix[i,:].T - \
                    datamatrix[j,:] * datamatrix[j,:].T

                if eta >= 0:
                    print 'eta >= 0'
                    continue

                alphas[j] -= label_mat[j] * (e_i - e_j) / eta
                alphas[j] = clip(alphas[j], high, low)

                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print 'j not moving enough'
                    continue

                alphas[i] += label_mat[j] * label_mat[i] * \
                    (alpha_j_old - alphas[j])

                b1 = b - e_i - label_mat[i] * (alphas[i] - alpha_i_old) * \
                    datamatrix[i,:] * datamatrix[i,:].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * \
                    datamatrix[i,:] * datamatrix[j,:].T

                b2 = b - e_j - label_mat[i] * (alphas[i] - alpha_i_old) * \
                    datamatrix[i,:] * datamatrix[j,:].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * \
                    datamatrix[j,:] * datamatrix[j,:].T

                if alphas[i] > 0 and c > alphas[i]:
                    b = b1
                elif alphas[i] > 0 and c > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alpha_pairs_changed += 1
                print 'iteration: %d i: %d, pairs changed: %d' % \
                    (iteration, i, alpha_pairs_changed)

        iteration = (iteration + 1) if alpha_pairs_changed == 0 else 0
        print 'iteration: %d' % iteration
    return b, alphas
