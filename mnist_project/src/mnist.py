import idx2numpy
import os
import numpy as np

def load_idx_mnist():

    DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')

    X_train = idx2numpy.convert_from_file(os.path.join(DATA_PATH, 'train-images.idx3-ubyte'))
    y_train = idx2numpy.convert_from_file(os.path.join(DATA_PATH, 'train-labels.idx1-ubyte'))
    X_test = idx2numpy.convert_from_file(os.path.join(DATA_PATH, 't10k-images.idx3-ubyte'))
    y_test = idx2numpy.convert_from_file(os.path.join(DATA_PATH, 't10k-labels.idx1-ubyte'))

    # Flattening Input 
    X_train = X_train.reshape(-1, 28*28) 
    X_test = X_test.reshape(-1, 28*28) 

    # Normalizing Input Data
    X_train = X_train.astype(np.float32)/255.0
    X_test = X_train.astype(np.float32)/255.0

    # One Hot Encoding Output Data
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test
