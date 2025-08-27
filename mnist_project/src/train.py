import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from src.mnist import load_idx_mnist
from src.utils import cross_entropy_loss
from src.model import Artificial_Neural_Network

def get_mini_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(0, X.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        yield X_batch, y_batch

def train_network(epochs = 10, batch_size = 64, learning_rate = 0.001):
    X_train, y_train, X_test, y_test = load_idx_mnist()
    
    input_size = X_train.shape[1]
    hiddenL_1 = 128
    hiddenL_2 = 64
    outputL = y_train.shape[1]

    ann = Artificial_Neural_Network(
        input_size, hiddenL_1, hiddenL_2, outputL, learning_rate=learning_rate
    )

    epoch_losses = []
    for epoch in range(epochs):
        batch_losses =[]
        for X_batch, y_batch in get_mini_batches(X_train,y_train, batch_size):
            y_hat = ann.train_step(X_batch, y_batch)
            batch_loss = cross_entropy_loss(y_batch, y_hat)
            batch_losses.append(batch_loss)

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), epoch_losses, marker='o', color='blue')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("✅ Loss X Epochs Graph as training_loss.png")


    y_pred = ann.forward_prop(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("MNIST Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion matrix heatmap saved as confusion_matrix.png")

    return ann


