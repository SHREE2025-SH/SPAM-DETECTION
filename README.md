# Spam Detection

A machine learning project that classifies SMS/email messages as **spam** or **ham** (not spam) using two models — Naive Bayes and Decision Tree — trained on a TF-IDF feature matrix.

## Project Structure

```
spam-detection/
│
├── spam.csv                    # Raw dataset (SMS Spam Collection)
├── spam_detection.py           # Training script: cleans data, trains & saves models
├── app.py                      # Streamlit web app for interactive predictions
│
├── models/
│   ├── naive_bayes_spam.pkl    # Trained Naive Bayes model
│   ├── decision_tree_spam.pkl  # Trained Decision Tree model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│
├── spam_detection_analysis.png # Visualizations generated after training
├── requirements.txt
└── README.md
```

## Models

| Model         | Notes                                   |
|---------------|-----------------------------------------|
| Naive Bayes   | Multinomial NB — fast and effective for text |
| Decision Tree | `max_depth=10` — interpretable baseline |

Both models are trained on **TF-IDF vectors** (top 3,000 features) with an 80/20 stratified train/test split.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the models

Run the training script to clean the data, train both classifiers, generate visualisations,
and save the model files to `models/`:

```bash
python spam_detection.py
```

### 3. Run the web app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Dataset

The project uses the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (`spam.csv`).  
Columns used: `v1` (label: `spam`/`ham`) and `v2` (message text).

## Text Preprocessing

1. Lowercase the message  
2. Remove punctuation  
3. Strip extra whitespace  

## Requirements

See [requirements.txt](requirements.txt).
