import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.feature_engineering import load_processed_data
from src.config import *

def evaluate():
    model, vectorizer = joblib.load(MODEL_PATH)

    X, y = load_processed_data()
    X_vec = vectorizer.transform(X)

    # same split logic
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # write metrics
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    print("Accuracy:", acc)

    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved")

if __name__ == "__main__":
    evaluate()

