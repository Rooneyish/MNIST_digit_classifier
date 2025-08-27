import numpy as np
from src.utils import softmax, leaky_relu, derivative_leaky_relu, cross_entropy_loss

class Artificial_Neural_Network:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate = 0.01, beta = 0.99, epsilon = 1e-10):
        self.W1 = np.random.randn(input_size,hidden1_size)
        self.b1 = np.zeros((1,hidden1_size))
        self.W2 = np.random.randn(hidden1_size,hidden2_size)
        self.b2 = np.zeros((1,hidden2_size))
        self.W3 = np.random.randn(hidden2_size,output_size)
        self.b3 = np.zeros((1,output_size))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta = beta 
        self.v_dw1 = np.zeros_like(self.W1)
        self.v_db1 = np.zeros_like(self.b1)
        self.v_dw2 = np.zeros_like(self.W2)
        self.v_db2 = np.zeros_like(self.b2)
        self.v_dw3 = np.zeros_like(self.W3)
        self.v_db3 = np.zeros_like(self.b3)

    
    def forward_prop(self, X):
        self.Z1 = np.dot(X,self.W1) + self.b1
        self.A1 = leaky_relu(self.Z1)
        self.Z2 = np.dot(self.A1,self.W2) + self.b2
        self.A2 = leaky_relu(self.Z2)
        self.Z3 = np.dot(self.A2,self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3
    
    def backward_prop(self, X, y, y_hat):
        m = X.shape[0]

        dZ3 = y_hat-y
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, dW3.T)
        dZ2 = dA2 * derivative_leaky_relu(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * derivative_leaky_relu(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.v_dw1 = self.beta * self.v_dw1 + (1-self.beta) * (dW1 ** 2)
        self.v_db1 = self.beta * self.v_db1 + (1-self.beta) * (db1 ** 2)
        self.v_dw2 = self.beta * self.v_dw2 + (1-self.beta) * (dW2 ** 2)
        self.v_db2 = self.beta * self.v_db2 + (1-self.beta) * (db2 ** 2)
        self.v_dw3 = self.beta * self.v_dw3 + (1-self.beta) * (dW3 ** 2)
        self.v_db3 = self.beta * self.v_db3 + (1-self.beta) * (db3 ** 2)

        self.W1 -= (self.learning_rate * dW1 )/(np.sqrt(self.v_dw1 + self.epsilon))
        self.b1 -= (self.learning_rate * db1 )/(np.sqrt(self.v_db1 + self.epsilon))
        self.W2 -= (self.learning_rate * dW2 )/(np.sqrt(self.v_dw2 + self.epsilon))
        self.b2 -= (self.learning_rate * db2 )/(np.sqrt(self.v_db2 + self.epsilon))
        self.W3 -= (self.learning_rate * dW3 )/(np.sqrt(self.v_dw3 + self.epsilon))
        self.b3 -= (self.learning_rate * db3 )/(np.sqrt(self.v_db3 + self.epsilon))

    def train_step(self, X, y):
        y_hat = self.forward_prop(X)
        self.backward_prop(X, y, y_hat)
        return y_hat


