import matplotlib.pyplot as plt

######################################################################
##### Tree plotting functions from Chapter 3 of
##### "Machine Learning in Action".
######################################################################

class PlotTreeOptions:
    """using an option class to replace plot_tree() attributes.
    Note that plot_tree() updates the xoff and yoff members, so we
    can't easily use an immutable object"""
    def __init__(self, totalw, totald, xoff, yoff):
        self.totalw = totalw
        self.totald = totald
        self.xoff = xoff
        self.yoff = yoff

decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plot_node(axes, node_txt, center_pt, parent_pt, node_type):
    axes.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                  xytext=center_pt, textcoords='axes fraction',
                  va='center', ha='center', bbox=node_type,
                  arrowprops=arrow_args)

def create_plot0():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axes = plt.subplot(111, frameon=False)
    plot_node(axes, 'a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(axes, 'a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()

def get_num_leafs(mytree):
    num_leafs = 0
    first_str = mytree.keys()[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(mytree):
    max_depth = 0
    first_str = mytree.keys()[0]
    second_dict = mytree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1

        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no',
                                       1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no',
                                       1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}},
                                                        1: 'no'}}}}
        ]
    return list_of_trees[i]

def plot_mid_text(axes, center_pt, parent_pt, text):
    xmid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    ymid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    axes.text(xmid, ymid, text)

def plot_tree(axes, mytree, parent_pt, node_text, options):
    num_leafs = get_num_leafs(mytree)
    first_str = mytree.keys()[0]
    center_pt = (options.xoff + (1.0 + float(num_leafs)) / 2.0 / options.totalw,
                 options.yoff)
    plot_mid_text(axes, center_pt, parent_pt, node_text)
    plot_node(axes, first_str, center_pt, parent_pt, decision_node)
    second_dict = mytree[first_str]
    options.yoff = options.yoff - 1.0 / options.totald
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(axes, second_dict[key], center_pt, str(key), options)
        else:
            options.xoff = options.xoff + 1.0 / options.totalw
            plot_node(axes, second_dict[key], (options.xoff, options.yoff),
                      center_pt, leaf_node)
            plot_mid_text(axes, (options.xoff, options.yoff), center_pt, str(key))
    options.yoff = options.yoff + 1.0 / options.totald

def create_plot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    axes = plt.subplot(111, frameon=False, **axprops)

    totalw = float(get_num_leafs(intree))
    options = PlotTreeOptions(totalw, float(get_tree_depth(intree)),
                              -0.5 / totalw, 1.0)
    plot_tree(axes, intree, (0.5, 1.0), '', options)
    plt.show()
