import numpy as np
from src.utils import softmax, leaky_relu, derivative_leaky_relu, cross_entropy_loss

class Artificial_Neural_Network:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate = 0.01):
        self.W1 = np.random.randn(input_size,hidden1_size)
        self.b1 = np.zeros((1,hidden1_size))
        self.W2 = np.random.randn(hidden1_size,hidden2_size)
        self.b2 = np.zeros((1,hidden2_size))
        self.W3 = np.random.randn(hidden2_size,output_size)
        self.b3 = np.zeros((1,output_size))
        self.learning_rate = learning_rate
    
    def forward_prop(self, X):
        self.Z1 = np.dot(X,self.W1) + self.b1
        self.A1 = leaky_relu(self.Z1)
        self.Z2 = np.dot(X,self.W2) + self.b2
        self.A2 = leaky_relu(self.Z2)
        self.Z3 = np.dot(X,self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3
    
        