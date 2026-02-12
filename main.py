from src.data_preprocessing import run_preprocessing
from src.train import train_model
from src.evaluate import evaluate

def main():
    run_preprocessing()
    train_model()
    evaluate()

if __name__ == "__main__":
    main()

