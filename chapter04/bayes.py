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
