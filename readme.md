# News Article Classification — ML Pipeline

## Project Overview:
  This project implements an end-to-end machine learning pipeline for news article category classification using Python scripts only. 
  The system preprocesses text data, converts it into TF-IDF features, trains a Linear SVM classifier, and evaluates performance using accuracy and confusion matrix metrics. 
  The full workflow runs from the terminal through a single entry script.

  ## Objective:
  Build a reproducible ML pipeline that classifies news articles into categories such as:
     Sports
     Business
     Technology
     Politics
     Entertainment
  The implementation follows strict script-only architecture without Jupyter notebooks.

  ## Dataset Source:
  - Dataset: BBC News style labeled dataset ( from kaggle)
  - File used: BBCNews.csv

Dataset contains:
descr → news article text
tags → comma-separated tags (first tag used as primary label)

During preprocessing:
first tag is extracted as class label
rare labels removed
only core news categories retained

## Model Choice:
Model: Linear Support Vector Machine (LinearSVC)

Rationale:
Performs strongly on sparse TF-IDF text features
Efficient for high-dimensional NLP data
Deterministic and stable for small/medium datasets
Lower overfitting risk vs complex models

## Feature Engineering
TF-IDF vectorization with:
n-grams (1–3)
max feature cap
rare term filtering
common term filtering
sublinear term frequency scaling
This improves semantic signal and margin separation.

## Preprocessing Steps
Missing value removal
Index column drop
Label normalization
Multi-tag → primary tag extraction
Rare class filtering
Core category filtering
Lowercasing
Symbol removal
Stopword removal
Short token filtering

## Class Imbalance Handling
Minimum class size threshold applied
Stratified train/test split
SVM uses class_weight="balanced"

  ## Hyperparameter Tuning
  GridSearchCV used for SVM tuning:
  Best parameter selected automatically.
  
  ## Evaluation Metrics
  Accuracy
  recall
  precision
  Confusion Matrix
  classification report
  Metrics saved to file
  Confusion matrix plot generated
    
  ## Final Result
        Classification Report:
                   precision    recall  f1-score   support
    
         business       0.91      0.93      0.92        80
    entertainment       0.99      0.94      0.96        83
         politics       0.92      0.95      0.93        59
           sports       0.99      0.98      0.98        95
       technology       0.97      1.00      0.99        78

       
         accuracy                           0.96       395
        macro avg       0.96      0.96      0.96       395
     weighted avg       0.96      0.96      0.96       395


## Project Structure
    news_classification_project/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   ├── train.py
    │   ├── evaluate.py
    │
    ├── models/
    │
    ├── results/
    │   └── metrics.txt
    │
    ├── requirements.txt
    ├── README.md
    └── main.py

## How To Run (Terminal Only)
1. Create Virtual Environment:

       python -m venv venv
        venv\Scripts\activate

3. Install Dependencies:
 
       pip install -r requirements.txt

4. Run Full Pipeline:

       python main.py

## Key Design Decisions
Script-only architecture (assignment compliant)
Modular pipeline design
Config-driven paths
Reproducible splits
Tuned linear SVM baseline
Feature-rich TF-IDF setup
Label noise reduction
Controlled class filtering

## Limitations:
 - Uses classical ML (no deep learning embeddings)
 - Label derived from first tag only
 - No semantic embeddings (by assignment constraint)





    

     
