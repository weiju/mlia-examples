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

            # check for violoation of KKT conditions
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
                alphas[j] = clip_alpha(alphas[j], high, low)  # note that numpy has clip()

                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print 'j not moving enough'
                    continue

                # update i by same amount as j, update is in opposite direction
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
                elif alphas[j] > 0 and c > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alpha_pairs_changed += 1
                print 'iteration: %d i: %d, pairs changed: %d' % \
                    (iteration, i, alpha_pairs_changed)

        iteration = (iteration + 1) if alpha_pairs_changed == 0 else 0
        print 'iteration: %d' % iteration
    return b, alphas


######################################################################
#### Full Platt SMO
######################################################################


class OptStruct:
    def __init__(self, datamat_in, class_labels, c, toler):
        self.x = datamat_in
        self.label_mat = class_labels
        self.c = c
        self.tol = toler
        self.m = shape(datamat_in)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.ecache = mat(zeros((self.m, 2)))


def calc_ek(o_s, k):
    f_xk = float(multiply(o_s.alphas, o_s.label_mat).T * \
                     (o_s.x * o_s.x[k,:].T)) + o_s.b
    e_k = f_xk - float(o_s.label_mat[k])
    return e_k

def select_j(i, o_s, e_i):
    max_k = -1
    max_delta_e = 0
    e_j = 0
    o_s.ecache[i] = [1, e_i]
    valid_ecache_list = nonzero(o_s.ecache[:,0].A)[0]
    if len(valid_ecache_list) > 1:
        for k in valid_ecache_list:
            if k == i:
                continue
            e_k = calc_ek(o_s, k)
            delta_e = abs(e_i - e_k)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                e_j = e_k
        return max_k, e_j
    else:
        j = select_j_rand(i, o_s.m)
        e_j = calc_ek(o_s, j)
    return j, e_j


def update_ek(o_s, k):
    o_s.ecache[k] = [1, calc_ek(o_s, k)]


def inner_l(i, o_s):
    e_i = calc_ek(o_s, i)
    if ((o_s.label_mat[i] * e_i < -o_s.tol) and (o_s.alphas[i] < o_s.c)) or \
            ((o_s.label_mat[i] * e_i > o_s.tol) and (o_s.alphas[i] > 0)):
        j, e_j = select_j(i, o_s, e_i)
        alpha_i_old = o_s.alphas[i].copy()
        alpha_j_old = o_s.alphas[j].copy()

        if o_s.label_mat[i] != o_s.label_mat[j]:
            low = max(0, o_s.alphas[j] - o_s.alphas[i])
            high = min(o_s.c, o_s.c + o_s.alphas[j] - o_s.alphas[i])
        else:
            low = max(0, o_s.alphas[j] + o_s.alphas[i] - o_s.c)
            high = min(o_s.c, o_s.alphas[j] + o_s.alphas[i])

        if low == high:
            print 'low == high'
            return 0

        eta = 2.0 * o_s.x[i,:] * o_s.x[j,:].T - o_s.x[i,:] * o_s.x[i,:].T - \
            o_s.x[j,:] * o_s.x[j,:].T

        if eta >= 0:
            print 'eta >= 0'
            return 0

        o_s.alphas[j] -= o_s.label_mat[j] * (e_i - e_j) / eta
        o_s.alphas[j] = clip_alpha(o_s.alphas[j], high, low)
        update_ek(o_s, j)

        if abs(o_s.alphas[j] - alpha_j_old) < 0.00001:
            print 'j not moving enough'
            return 0

        o_s.alphas[i] += o_s.label_mat[j] * o_s.label_mat[i] * \
            (alpha_j_old - o_s.alphas[j])
        update_ek(o_s, i)

        b1 = o_s.b - e_i - o_s.label_mat[i] * (o_s.alphas[i] - alpha_i_old) * \
            o_s.x[i,:] * o_s.x[i,:].T - o_s.label_mat[j] * \
            (o_s.alphas[j] - alpha_j_old) * o_s.x[i,:] * o_s.x[j,:].T
        b2 = o_s.b - e_j - o_s.label_mat[i] * (o_s.alphas[i] - alpha_i_old) * \
            o_s.x[i,:] * o_s.x[j,:].T - o_s.label_mat[j] * \
            (o_s.alphas[j] - alpha_j_old) * o_s.x[j,:] * o_s.x[j,:].T

        if o_s.alphas[i] > 0 and o_s.c > o_s.alphas[i]:
            o_s.b = b1
        elif o_s.alphas[j] > 0 and o_s.c > o_s.alphas[j]:
            o_s.b = b2
        else:
            o_s.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smo_p(datamat_in, class_labels, c, toler, max_iter, ktup=('lin', 0)):
    o_s = OptStruct(mat(datamat_in), mat(class_labels).transpose(), c, toler)
    iteration = 0
    entire_set = True
    alpha_pairs_changed = 0

    while iteration < max_iter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(o_s.m):
                alpha_pairs_changed += inner_l(i, o_s)
            print 'entire_set, iteration: %d i: %d, pairs changed: %d' % \
                (iteration, i, alpha_pairs_changed)
            iteration += 1
        else:
            non_bound_is = nonzero((o_s.alphas.A > 0) * (o_s.alphas.A < c))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_l(i, o_s)
                print 'non-bound, iteration: %d, i: %d, pairs changed: %d' % \
                    (iteration, i, alpha_pairs_changed)
            iteration += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print 'iteration number: %d' % iteration
    return o_s.b, o_s.alphas


def calc_ws(alphas, data_arr, class_labels):
    x = mat(data_arr)
    label_mat = mat(class_labels).transpose()
    m, n = shape(x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], x[i,:].T)
    return w


