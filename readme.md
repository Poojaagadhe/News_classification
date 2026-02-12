# News Article Classification — ML Pipeline

## Project Overview:
 This project implements a script-based machine learning pipeline to classify news articles into major categories using NLP techniques. 
 The system performs text preprocessing, TF-IDF feature extraction, and Linear SVM training with hyperparameter tuning. 
 The full workflow runs from the terminal through modular Python scripts.

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
  - Format: CSV
  - Text column: descr
  - Label column: tags
  - First tag is extracted as the primary class label
  - Rare classes are filtered to avoid class sparsity and split errors

During preprocessing:
first tag is extracted as class label
rare labels removed
only core news categories retained

## Model Choice:
  - Linear Support Vector Machine (Linear SVM)
  - Suitable for high-dimensional sparse text features
  - Works effectively with TF-IDF vectors
  - Uses class balancing
  - Hyperparameter tuning performed using GridSearch cross-validation

## Feature Engineering
TF-IDF vectorization with:
n-grams (1–3)
max feature cap
rare term filtering
common term filtering
sublinear term frequency scaling
This improves semantic signal and margin separation.

  ## Evaluation Metrics
  Accuracy
  recall
  precision
  Confusion Matrix
  classification report
  Metrics saved to file
  Confusion matrix plot generated
    
  ## Final Result
  - Text cleaned and normalized
  - TF-IDF n-gram features generated
  - Linear SVM trained with tuned C parameter
  - Metrics saved in results/metrics.txt for review

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





    

     

