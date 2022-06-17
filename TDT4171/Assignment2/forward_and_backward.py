import numpy as np

def printing_function(x, name):
    for i in range(len(x)):
        print(name, round(x[i][0],3), round(x[i][1],3))

def print_func(forward, backward, smoothing):
    printing_function(forward, 'Forward')
    print('')
    printing_function(backward, 'Backward')
    print('')
    printing_function(smoothing, 'Smoothing')
    print('')

# Transition matrix
transition = np.array([[0.7, 0.3],[0.3,0.7]]) 

# Emission matrix:
emission = {True: np.array([0.9, 0.2]),
            False: np.array([0.1, 0.8])}

P_initial = np.array([0.5,0.5]) # initial probability for rain

def normalize(xx):
    return xx / np.sum(xx)

def forward_alg(forward_vector, i):
    return normalize(emission[i] * (transition @ forward_vector))
    
def backward_alg(backward_vector, i):
    return (transition * emission[i]) @ backward_vector

def smoothing(forward_vector, backward_vector):
    smoothing = np.multiply(forward_vector,backward_vector)
    return normalize(smoothing)


def forward_backward(P_initial, evidence, printing = True):
    forward_vector, backward_vector, smoothing_vector = [], [], []
    forward_vector.append(P_initial)
    backward = np.array([1,1])
    N = len(evidence)

    for i in range(0,N):
        forward_vector.append(forward_alg(forward_vector[i], evidence[i]))
    
    for i in range(N, 0, -1):
        smoothing_vector.append(smoothing(forward_vector[i], backward))
        backward = backward_alg(backward, evidence[i-1])
        backward_vector.append(backward)
    
    smoothing_vector.reverse()
    print_func(forward_vector[1:], backward_vector, smoothing_vector)

evidence_test = np.array([True,True])
forward_backward(P_initial, evidence_test)

evidence = np.array([True,True,False,True,True])
forward_backward(P_initial, evidence)

evidence = np.array([False,False,True,True,True,False]) 
forward_backward(P_initial, evidence)


    





