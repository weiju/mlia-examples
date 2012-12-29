from numpy import *

def load_dataset():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classes = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, classes

def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word '%s' is not in my vocabulary !" % word
    return return_vec

def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = log(p1_num / p1_denom)  # change to log()
    p0_vect = log(p0_num / p0_denom)  # change to log()
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    return 1 if p1 > p0 else 0

def testing_nb():
    posts, classes = load_dataset()
    vocabs = create_vocab_list(posts)
    train_mat = []
    for post in posts:
        train_mat.append(set_of_words2vec(vocabs, post))
    p0v, p1v, pab = train_nb0(array(train_mat), array(classes))
    test_entry = ['love', 'my', 'dalmatian']
    doc = array(set_of_words2vec(vocabs, test_entry))
    print test_entry, 'classified as: ', classify_nb(doc, p0v, p1v, pab)

    test_entry = ['stupid', 'garbage']
    doc = array(set_of_words2vec(vocabs, test_entry))
    print test_entry, 'classified as: ', classify_nb(doc, p0v, p1v, pab)

def bag_of_words2vec_mn(vocabs, input_set):
    result_vec = [0] * len(vocabs)
    for word in input_set:
        if words in input_set:
            if word in vocabs:
                result_vec[vocabs.index(word)] += 1
    return result_vec

