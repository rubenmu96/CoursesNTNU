'''
TDT4171 - Assignment 4
Student number: 480858

The code for assignment 4 consists of 3 files:
helper_functions.py
decision_tree.py
main.py

'''

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *
from decision_tree import *

# Load the data 
training_data = load_data('train.csv')
test_data = load_data('test.csv')

# Possible values
Values = [1,2]

# Attributes
attributes = [x for x in range(len(training_data[0])-1)]


importance_types = ['gain', 'gain_random', 'random']
for importance_type in importance_types:
    '''
    Generate the decision trees for the different importance types
    '''
    design_tree(training_data, test_data, Values, attributes, importance_type, importance_type)
    

def simulation(training_data, test_data, Values, attributes):
    '''
    Here we calculate the test accuracy 200 times to see how it changes
    (due to the randomness in random importance gain and random importance)
    and plot the test accuracies.
    '''
    test_accuracy_gain = []
    test_accuracy_random_gain = []
    test_accuracy_random = []
    
    for i in range(200):
        tree = decision_tree(training_data, attributes, None, Values, method = 'gain')
        test_accuracy_gain.append(accuracy(tree, test_data))
        
        tree = decision_tree(training_data, attributes, None, Values, method = 'gain_random')
        test_accuracy_random_gain.append(accuracy(tree, test_data))
        
        tree = decision_tree(training_data, attributes, None, Values, method = 'random')
        test_accuracy_random.append(accuracy(tree, test_data))

    plt.figure(figsize = (20,5))
    plt.subplot(1,3,1)
    plt.title('Information gain')
    plt.plot(test_accuracy_gain)
    plt.subplot(1,3,2)
    plt.title('Random information gain')
    plt.plot(test_accuracy_random_gain)
    plt.subplot(1,3,3)
    plt.title('Random number')
    plt.plot(test_accuracy_random)
    plt.show()
    return test_accuracy_gain, test_accuracy_random_gain, test_accuracy_random

test_g, test_rg, test_r = simulation(training_data, test_data, Values, attributes)


print('Average test accuracy using random information gain ' + str(np.round(np.mean(test_rg),3)) + ' +/-' + str(np.round(np.std(test_rg),3)))
print('Average test accuracy using random importance ' + str(np.round(np.mean(test_r),3)) + '+/-' + str(np.round(np.std(test_r),3)))
