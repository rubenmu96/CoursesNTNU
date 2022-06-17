import numpy as np
import random
import pandas as pd
import math
import graphviz as gv

# TODO
# 1. set seed
# 2. implement random importance => what random values? check algorithm
# 3. implement information gain
# 4. Plot network using Graphviz


# Load training and test set:
### can print each column using df['x'] where x = 1, 1.1, 2, 2.1, 1.2, 1.3, 1.4, 1.5

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')



probs = train_df.value_counts(normalize = True)


class Tree():
    def __init__(self, attribute):
        self.children = {}
        self.attribute = attribute
        self.split_value = None
    def add_child(self, edge, child):
        self.children[edge] = child
    def add_split_value(self, split_value):
        self.split_value = split_value

import random
import math
import graphviz as gv


class Node:     # Nodes in the tree as class
    def __init__(self, data, leaf):
        self.data = data
        self.children = {}
        self.leaf = leaf

    def add_child(self, key, value):
        self.children[key] = value


def B(q):   # Formula for entropy
    if q == 1 or q == 0:
        return 0
    return -(q * math.log(q, 2) + (1 - q) * math.log(1 - q, 2))


def argmax(examples, attributes, rand):
    if rand:    # Choosing the attribute randomly
        return random.choice(attributes)
    else:       # Choosing the attribute using information gain
        gain = []
        for a in attributes:
            gain.append(importance(a, examples))
        return attributes[gain.index(max(gain))]


def importance(a, examples):
    # Importance function for finding argmax
    # E = [E0, E1] = [[p0, n0], [p1, n1]]
    # p = 1, n = 2
    E = [[0.0, 0.0], [0.0, 0.0]]
    for example in examples:
        if example[a] == 1 and example[-1] == 1:
            E[0][0] += 1
        elif example[a] == 1 and example[-1] == 2:
            E[0][1] += 1
        elif example[a] == 2 and example[-1] == 1:
            E[1][0] += 1
        elif example[a] == 2 and example[-1] == 2:
            E[1][1] += 1
    p = E[0][0] + E[1][0]
    n = E[0][1] + E[1][1]
    remainder = 0
    for j in range(2):
        remainder += (E[j][0] + E[j][1])/(p + n) * B(E[j][0]/(E[j][0]+E[j][1]))
    return B(p/(p+n)) - remainder


def plurality_value(a):
    # Plurality function for selecting most common output value
    return max(set([x[-1] for x in a]), key=a.count)


def decision_tree_learning(examples, attributes, parent_examples, rand):
    # Creating a list to use with plurality function
    output = []
    for example in examples:
        output.append(example[-1])
    if not examples:
        return Node(plurality_value(parent_examples), True)
    elif len(set(output)) == 1:
        return Node(set(output).pop(), True)
    elif not attributes:
        return Node(plurality_value(output), True)
    else:
        A = argmax(examples, attributes, rand)
        root = Node(A, False)
        for vk in range(1, 3):
            exs = []
            for example in examples:
                if example[A] == vk:
                    exs.append(example)
            copy = list(attributes)
            copy.remove(A)
            subtree = decision_tree_learning(exs, copy, examples, rand)
            root.add_child(vk, subtree)
    return root


def readfile(filename):
    # Reading the training set and creating a list
    with open(filename, "r") as f:
        data = f.readlines()
    datalist = []
    for line in data:
        datalist.append([int(i) for i in line.split()])
    return datalist


def count(tree, test):
    while not tree.leaf:
        if test[tree.data] == 1:
            tree = tree.children[1]
        else:
            tree = tree.children[2]
    return tree.data


def draw(g, tree, parent):
    if not tree.leaf:
        g.node(repr(tree), str(tree.data))
        draw(g, tree.children[1], tree)
        draw(g, tree.children[2], tree)
    else:
        g.node(repr(tree), str(tree.data))
    if parent is not None:
        g.edge(repr(parent), repr(tree))
    return g


def main():
    # Loading the training set
    examples = readfile("data/training.txt")
    attributes = [x for x in range(len(examples[0])-1)]
    # Training the decision tree
    tree = decision_tree_learning(examples, attributes, None, False)
    # Loading the test file
    test = readfile("data/test.txt")
    counter = 0
    # Counter for counting how many decisions are correct.
    for line in test:
        if count(tree, line) == line[len(line)-1]:
            counter += 1
    g = gv.Digraph(format='svg')
    g = draw(g, tree, None)
    g.render('g')
main()
