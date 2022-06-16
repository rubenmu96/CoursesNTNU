import numpy as np 
import pandas as pd 

class LogisticRegression:
    def __init__(self, iters, gamma, lambda_ = 0):
        # learning rate
        self.gamma = gamma
        # How many times to run gradient descent
        self.iters = iters
        # Regularization parameter
        self.lambda_ = lambda_

    def fit(self, X, y):
        Xb = np.column_stack((np.ones(X.shape[0]), X)) # add bias term
        
        # weight of size 3 (one weight for each (bias, x0 and x1))
        self.w = np.random.randn(Xb.shape[1])
        
        # We update the weight values using gradient descent (+ regularization)
        for i in range(self.iters):
            self.w += - self.gamma * np.dot(sigmoid(np.dot(Xb,self.w))-y, Xb)/y.size + self.lambda_*self.w/y.size

    def predict(self, X):
        Xb = np.column_stack((np.ones(X.shape[0]), X)) # add bias term
        
        # Get prediction from the Sigmoid function
        y_pred = sigmoid(np.dot(Xb,self.w))
        return y_pred

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    return 1. / (1. + np.exp(-x))