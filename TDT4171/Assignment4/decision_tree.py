'''
TDT4171 - Assignment 4
Student number: 480858

The code for assignment 4 consists of 3 files:
helper_functions.py
decision_tree.py
main.py

'''

# This might be necessary when running code (?)
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

import numpy as np
import graphviz as gv
from helper_functions import *

class build_tree:
    def __init__(self, data):
        self.data = data
        self.children = {}
         
def decision_tree(examples, attributes, parents, Values, method):
    '''
    Algorithm found in the book
    
    Y: the value we are trying to predict (which will be the last column in examples)
    
    It supports 3 different methods
    1. Gain (where the first attribute gets chosen if there are more than 1 that shares the same gain)
    2. Information gain with random choosing of attribute if there are multiple values with same gain
    3. Random importance: Each attributes have been assigned a random attribute
    '''

    # Get the Y values (or class values)
    if len(examples) == 1:
        Y = [examples[0][-1]]
    elif len(examples) == 0: 
        return build_tree(PLURALITY(parents)) 
    else:
        Y = np.array(examples)[:,-1]

    if is_unique(Y): # check if all the attributes have the same classification
        return build_tree(Y[0])
    elif not attributes: # check if attributes is empty 
        return build_tree(PLURALITY(Y))
    else:
        A = IMPORTANCE(examples, attributes, method)
        root_node = build_tree(A)

        # Remove the attribute after it has been picked
        attributes = list(attributes)
        attributes.remove(A)

        for value in Values:
            examples_next = []
            for e in examples:
                if e[A] == value:
                    examples_next.append(e)

            subtree = decision_tree(examples_next, attributes, examples, Values, method)
            root_node.children[value] = subtree
    return root_node

def design_tree(training_data, test_data, Values, attributes, method, save_file):
    '''
    This function designs the tree and calculate the accuracies of t.he training and test set.
    
    It supports 3 different methods
    1. Gain (where the first attribute gets chosen if there are more than 1 that shares the same gain)
    2. Information gain with random choosing of attribute if there are multiple values with same gain
    3. Random importance: Each attributes have been assigned a random attribute
    '''
    def make_tree(DT, tree):
        edges = []
        for edge, value in tree.children.items():
            edges.append(edge)
        if tree.children:
            DT.node(repr(tree), str(tree.data))
            make_tree(DT, tree.children[1])
            make_tree(DT, tree.children[2])
            if tree.children:
                DT.edge(repr(tree), repr(tree.children[1]), label = str(edges[0]))
                DT.edge(repr(tree), repr(tree.children[2]), label = str(edges[1]))
        else:
            DT.node(repr(tree), str(tree.data), color = 'blue')
        return DT
    
    
    if method == 'gain':
        print('Information gain')
        tree = decision_tree(training_data, attributes, None, Values, method = 'gain')
        print('------------------------ TRAIN DATA ------------------------')
        accuracy(tree, training_data, True)
        print('------------------------ TEST  DATA ------------------------')
        accuracy(tree, test_data, True)
        print('')
    elif method == 'gain_random':
        print('Random information gain')
        tree = decision_tree(training_data, attributes, None, Values, method = 'gain_random')
        print('------------------------ TRAIN DATA ------------------------')
        accuracy(tree, training_data, True)
        print('------------------------ TEST  DATA ------------------------')
        accuracy(tree, test_data, True)
        print('')
    elif method == 'random':
        print('Random importance')
        tree = decision_tree(training_data, attributes, None, Values, method = 'random')
        print('------------------------ TRAIN DATA ------------------------')
        accuracy(tree, training_data, True)
        print('------------------------ TEST  DATA ------------------------')
        accuracy(tree, test_data, True)
        print('')
    else:
        print('Please specify an importance method: Either gain, gain_random or random.')
    DT = gv.Digraph(format='svg')
    DT = make_tree(DT, tree)
    DT.render(save_file)
