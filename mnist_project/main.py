from src.train import train_network

def main():
    trained_model = train_network(
        epochs=100,
        batch_size=128,
        learning_rate=0.0005
    )
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
