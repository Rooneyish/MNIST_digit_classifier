import numpy as np

def softmax(z):
    exp_z = np.exp(z-np.max(z, axis=1, keepdims= True))
    return exp_z/np.sum(exp_z, axis=1, keepdims= True)

def leaky_relu(z, alpha = 0.01):
    return np.maximum(alpha*z, z)

def derivative_leaky_relu(z, alpha = 0.01):
    return (z > 0).astype(float) + alpha * (z <= 0).astype(float)

def cross_entropy_loss(y, y_hat):
    epsilon = 1e-10
    loss = -np.sum(y*np.log(y_hat + epsilon))/y.shape[0]
    return loss

