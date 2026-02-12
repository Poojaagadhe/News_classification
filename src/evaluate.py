import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import train_test_split

from src.feature_engineering import load_processed_data
from src.config import *


def evaluate():

    # ensure results folder exists
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    # load model bundle
    model, vectorizer = joblib.load(MODEL_PATH)

    # load data
    X, y = load_processed_data()
    X_vec = vectorizer.transform(X)

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # predictions
    preds = model.predict(X_test)

    # accuracy
    acc = accuracy_score(y_test, preds)

    # confusion matrix
    cm = confusion_matrix(y_test, preds)

    # classification report dict + text
    report_dict = classification_report(
        y_test, preds, output_dict=True
    )
    report_text = classification_report(y_test, preds)

    # -------------------------
    # write metrics.txt
    # -------------------------
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("Accuracy:", acc)

    # -------------------------
    # save per-class metrics CSV
    # -------------------------
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv("results/classification_report.csv")

    # keep only class rows for plotting
    class_rows = df_report.iloc[:-3][
        ["precision", "recall", "f1-score"]
    ]

    # -------------------------
    # bar chart per-class metrics
    # -------------------------
    ax = class_rows.plot(kind="bar")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/per_class_metrics.png")
    plt.close()

    # -------------------------
    # confusion matrix plot
    # -------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # -------------------------
    # misclassification analysis
    # -------------------------
    mis_mask = preds != y_test.values
    mis_df = pd.DataFrame({
        "true": y_test.values[mis_mask],
        "pred": preds[mis_mask]
    })

    mis_df.to_csv("results/misclassified_samples.csv", index=False)

    print("Saved metrics, plots, and misclassification file")


if __name__ == "__main__":
    evaluate()




