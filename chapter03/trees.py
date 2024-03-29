import math
import operator
import pickle


def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


def shannon_entropy(dataset):
    num_entries = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        unique_vals = set([example[i] for example in dataset])
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            p = len(sub_dataset) / float(len(dataset))
            new_entropy += p * shannon_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(classes):
    class_count = {}
    for vote in classes:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    """Notes:
    1. dataset can't be 0-sized !
    2. There is a nasty side-effect in this function: it deletes elements out of its labels
       argument, needs rework. Due to its recursive nature, the labels parameter needs
       to shrink, but that does not mean that a top-level call has to deal with it
    """
    classes = [example[-1] for example in dataset]

    def same_class():
        return classes.count(classes[0]) == len(classes)

    def no_more_features():
        return len(dataset[0]) == 1

    if same_class():
        return classes[0]
    if no_more_features():
        return majority_count(classes)

    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    tree = {best_feature_label: {}}
    del(labels[best_feature])
    for value in set([example[best_feature] for example in dataset]):
        sub_labels = labels[:]
        tree[best_feature_label][value] = create_tree(split_dataset(dataset,
                                                                    best_feature,
                                                                    value),
                                                      sub_labels)
    return tree

    
def classify(input_tree, feat_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    with open(filename, 'w') as outfile:
        pickle.dump(input_tree, outfile)


def grab_tree(filename):
    with open(filename) as infile:
        return pickle.load(infile)
