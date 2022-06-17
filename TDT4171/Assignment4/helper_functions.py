'''
TDT4171 - Assignment 4
Student number: 480858

The code for assignment 4 consists of 3 files:
helper_functions.py
decision_tree.py
main.py
'''

import numpy as np
import random
import pandas as pd

def load_data(PATH):
    '''
    Load data from given path
    '''
    data = pd.read_csv(PATH)
    data_fixed = data.values.tolist()
    return data_fixed
        
def positive(examples):
    '''
    Calculate the number of positive samples
    '''
    positive_samples = 0
    examples = np.array(examples)
    last_column = examples[:,-1]
    
    for element in last_column:
        if element == 1:
            positive_samples += 1
    return positive_samples

def Binary_entropy(p):
    '''
    Calculate the binary entropy using the equation found in the book:
    - (p*log2(p) + (1-p)*log2(1-q))
    '''
    if p in [0,1]:
        return 0
    elif 1 - p < 0:
        return - (p*np.log2(p) + (1 - p) * np.log2(p - 1))
    else:
        return - (p*np.log2(p) + (1 - p) * np.log2(1 - p))
    
def importance_gain(examples, attribute):
    '''

    Calculate the importance using information gain for an attribute i.

    Here we use the equations that were given in the book (Choosing attribute tests)
    Gain(A) = B(p/(p+n)) - Remainder(A)
    where 
    Remainder(A) = sum of (p_k + n_k) / (p + n) B(p_k/(p_k + n_k))
    '''
    # Initialize 
    N = len(examples)
    remainder = 0
    
    samples_for_one = []
    samples_for_two = []
    Bvalues = [] # used to store values that are calculated using binary entropy

    # For attribute i, we divide it into 2 subsets 
    for e in examples:
        if e[attribute] == 1:
            samples_for_one.append(e)
        elif e[attribute] == 2:
            samples_for_two.append(e)

    # Calculate the B(q)
    positive_samples1 = positive(samples_for_one) # find the positive samples for subset 1
    Bvalues.append(Binary_entropy(positive_samples1 / len(samples_for_one)))
    
    positive_samples2 = positive(samples_for_two) # find the positive samples for subset 2
    Bvalues.append(Binary_entropy(positive_samples2 / len(samples_for_two)))

    # Find the positive samples of the whole dataset and calculate the B value
    posistive_samples_data = positive(examples)
    Bvalues.append(Binary_entropy(posistive_samples_data))
    
    # Calculate the remainder 
    remainder += len(samples_for_one) / N * Bvalues[0]
    remainder += len(samples_for_two) / N * Bvalues[1]


    return Bvalues[-1] - remainder

def importance_random(examples, attribute):
    '''
    Random importance aka choose the best attribute randomly by assigning some value, then choose the largest
    '''
    attribute_importance = []
    for i in attribute:
        attribute_importance.append(random.random())
    attribute_importance = np.array(attribute_importance)
    return attribute[attribute_importance.argmax()]

def IMPORTANCE(examples, attribute, method):
    '''
    Here we calculate the gain using 3 different methods
    1. Information gain. If two or more attributes have the same gain, it will choose the first attribute
    2. Random information gain: If two or more attributes have the same gain, 
    it will choose randomly between the attributes of which gets chosen
    3. Random: Gives each attribute a random importance, and chooses the one that has the highest.
    '''
    gains = []
    if method == 'gain':
        for a in attribute:
            gains.append(importance_gain(examples, a))
        max_gain_position = np.array(gains).argmax()
        return attribute[max_gain_position]
    elif method == 'gain_random':
        for a in attribute:
            gains.append(importance_gain(examples, a))
        max_gain_position = np.argwhere(gains == np.amax(gains)).flatten().tolist()
        return attribute[random.sample(max_gain_position,1)[0]]
    else:
        return importance_random(examples, attribute)
    
def is_unique(x):
    # Find if the element is unique or not
    first_element = x[0]
    for i in x[1:]:
        if i != first_element:
            return False
    return True
    
def PLURALITY(examples):
    '''
    Finds the value that occur the most.
    
    If both have the same number, we choose randomly.
    '''
    examples = np.array(examples)
    last_column = examples[:,-1]
    
    plurality = np.bincount(last_column)#.argmax()
    
    if plurality[1] == plurality[2]:
        return random.randint(1,2)
    else:
        return plurality.argmax()


def accuracy(tree, data, printing = False):
    '''
    Calculate the accuracy 
    '''
    N = len(data)
    
    correct, wrong = 0, 0
    def check_row(tree, row):
        while tree.children:
            tree = tree.children[row[tree.data]]
        return tree.data
    
    for row in data:
        if check_row(tree, row) == row[-1]:
            correct += 1
        else:
            wrong += 1
    if printing:
        print('The number of correct classifications are: ' 
              + str(correct) + '/' + str(N) + ' or ' + str(round(correct/N*100,2)) +'%')
        print('The number of wrong classifications are: ' 
              + str(wrong) + '/' + str(N) + ' or ' + str(round(wrong/N*100,2)) +'%')
    
    return correct / N
