import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

from src.feature_engineering import build_vectorizer, load_processed_data
from src.config import *

def train_model():
    X, y = load_processed_data()

    vectorizer = build_vectorizer()
    X_vec = vectorizer.fit_transform(X)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # keeps label balance
    )

    # base svm with imbalance handling
    svm = LinearSVC(class_weight="balanced")

    # hyperparameter grid
    param_grid = {
    "C": [0.25, 0.5, 1, 2, 4, 8]
    }


    # grid search tuning
    grid = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
    )


    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("Best SVM C:", grid.best_params_)

    # save model + vectorizer
    joblib.dump((best_model, vectorizer), MODEL_PATH)

    print("Tuned SVM model saved")

    return X_test, y_test

if __name__ == "__main__":
    train_model()

