# 📧 Email Spam Detection System

> **Classifying emails as Spam or Ham using NLP and Logistic Regression with TF-IDF vectorization.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)

---

## 📌 Project Overview

This project implements a complete **Natural Language Processing (NLP)** pipeline to detect spam emails. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert raw email text into numerical features, and **Logistic Regression** as the classification engine. The system is deployed as a real-time **Streamlit app** where users can paste any email text and instantly get a Spam/Ham verdict.

---

## 🧠 NLP Pipeline Architecture

```
email_spam.csv
       │
       ▼
  Text Cleaning          ◄── Combine: title + text
  (fillna, strip)
       │
       ▼
TfidfVectorizer           ◄── max_features=5000, stop_words='english'
  (Fit + Transform)                    │
       │                               ├── tfidf_vectorizer.pkl (saved)
       ▼
Logistic Regression       ◄── Train on TF-IDF features
   (Training)                          │
       │                               └── spam_model.pkl (saved)
       ▼
  Prediction Result
  (Spam / Ham)
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **File** | `email_spam.csv` |
| **Columns Used** | `title`, `text`, `type` (label) |
| **Label Classes** | `spam`, `ham` |
| **Train / Test Split** | 80% / 20% |

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| **Vectorizer** | TF-IDF Vectorizer |
| `max_features` | 5,000 top words |
| `stop_words` | English stop words removed |
| **Classifier** | Logistic Regression |
| **Model File** | `spam_model.pkl` |
| **Vectorizer File** | `tfidf_vectorizer.pkl` |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | `64.71%` |

> [!IMPORTANT]
> The 64.71% accuracy reflects the challenge of NLP on a dataset with diverse email styles and mixed languages. The model is trained to generalize on English content. Using `max_features=5000` ensures only the most informative tokens are used, reducing noise.

> [!TIP]
> To improve accuracy further, consider: (1) using a larger vocab size, (2) applying `MultinomialNB` or `SVM`, (3) adding n-gram features `(1,2)`, or (4) using a pre-trained BERT-based model for embeddings.

---

## 📉 Visualizations Generated

After running `train.py`, the following plots are saved:

| # | Output File | Description |
|---|---|---|
| 1 | `output_screenshot_confusion_matrix.png` | Heatmap showing true/false positives & negatives |
| 2 | `output_screenshot_distribution.png` | Pie chart of Spam vs Ham class balance |
| 3 | `output_screenshot_top_words.png` | Top 20 most defining words by TF-IDF score |

---

## 🖥️ Interactive Dashboard (`app.py`)

The Streamlit app allows users to:
- **Paste any email text** directly into the input box
- Get an **instant classification** verdict: `✅ Ham (Not Spam)` or `🚨 Spam`
- View confidence scores and keywords

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
Generates `spam_model.pkl`, `tfidf_vectorizer.pkl`, and visualization screenshots.

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
2_Email_Spam_Detection/
│
├── train.py                              # NLP training pipeline
├── app.py                                # Streamlit interactive classifier
├── email_spam.csv                        # Labeled email dataset
├── spam_model.pkl                        # Trained Logistic Regression model
├── tfidf_vectorizer.pkl                  # Fitted TF-IDF vectorizer
├── requirements.txt                      # Python dependencies
├── Model_Outputs_Record.md               # Detailed classification report
│
├── output_screenshot_confusion_matrix.png
├── output_screenshot_distribution.png
└── output_screenshot_top_words.png
```

---

## 🛠️ Dependencies

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---

## 🔗 Part of the Supervised Learning Projects Collection
← [Back to Main Repository](../README.md)
