from src.mnist import load_idx_mnist

def main():
    X_train, y_train, X_test, y_test = load_idx_mnist()
      
    print("\n Data Preprocessing Complete!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test  shape:", X_test.shape)
    print("y_test  shape:", y_test.shape)

    print("\nExample (first training sample):")
    print("Image vector (first 20 pixels):", X_train[0][:20])
    print("Label (one-hot):", y_train[0])

if __name__ == "__main__":
    main()
