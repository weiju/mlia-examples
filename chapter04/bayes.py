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


def bag_of_words2vec_mn(vocabs, input_set):
    result_vec = [0] * len(vocabs)
    for word in input_set:
        if word in vocabs:
            result_vec[vocabs.index(word)] += 1
    return result_vec


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

def text_parse(big_string):
    import re
    tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in tokens if len(tok) > 2]

def spam_test():
    docs = []
    classes = []
    full_text = []

    def add_doc_to_class(path, class_num):
        words = text_parse(open(path).read())
        docs.append(words)
        full_text.extend(words)
        classes.append(class_num)
        
    for i in range(1, 26):
        add_doc_to_class('email/spam/%d.txt' % i, 1)
        add_doc_to_class('email/ham/%d.txt' % i, 0)
    
    vocabs = create_vocab_list(docs)
    training_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []

    for doc_index in training_set:
        train_mat.append(set_of_words2vec(vocabs, docs[doc_index]))
        train_classes.append(classes[doc_index])

    p0v, p1v, pspam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0

    for doc_index in test_set:
        words = set_of_words2vec(vocabs, docs[doc_index])
        if classify_nb(array(words), p0v, p1v, pspam) != classes[doc_index]:
            error_count += 1
    print 'the error rate is: ', float(error_count) / len(test_set)

######################################################################
## RSS feed classification
######################################################################

def calc_most_freq(vocabs, full_text):
    import operator
    freqs = {}
    for token in vocabs:
        freqs[token] = full_text.count(token)
    return sorted(freqs.iteritems(), key=operator.itemgetter(1),
                  reverse=True)[:30]

def local_words(feed1, feed0):
    #import feedparser
    docs = []
    classes = []
    full_text = []
    def add_feed_to_class(feed, class_num):
        words = text_parse(feed['entries'][i]['summary'])
        docs.append(words)
        full_text.extend(words)
        classes.append(class_num)

    min_len = min(len(feed1['entries']), len(feed0['entries']))

    for i in range(min_len):
        add_feed_to_class(feed1, 1)
        add_feed_to_class(feed0, 0)

    vocabs = create_vocab_list(docs)
    top30_words = calc_most_freq(vocabs, full_text)
    for pair_w in top30_words:
        if pair_w[0] in vocabs:
            vocabs.remove(pair_w[0])

    training_set = range(2 * min_len)
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words2vec_mn(vocabs, docs[doc_index]))
        train_classes.append(classes[doc_index])

    p0v, p1v, pspam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:
        words = bag_of_words2vec_mn(vocabs, docs[doc_index])
        if classify_nb(array(words), p0v, p1v, pspam) != classes[doc_index]:
            error_count += 1
    print 'the error rate is: ', float(error_count) / len(test_set)
    return vocabs, p0v, p1v

def get_top_words(ny, sf):
    import operator
    vocabs, p0v, p1v = local_words(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            top_sf.append((vocabs[i], p0v[i]))
        if p1v[i] > -6.0:
            top_ny.append((vocabs[i], p1v[i]))
    
    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sorted_sf:
        print item[0]

    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY*"
    for item in sorted_ny:
        print item[0]
