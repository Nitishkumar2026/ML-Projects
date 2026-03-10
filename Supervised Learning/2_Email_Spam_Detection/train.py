import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("Loading Spam Dataset...")
df = pd.read_csv("email_spam.csv")

# Data cleaning
df['text'] = df['text'].fillna('')
df['content'] = df['title'].fillna('') + " " + df['text']

print("Preprocessing and Vectorizing...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['content'])
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression Model...")
model = LogisticRegression()
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save Model and Vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Generate 'Screenshots' of Model Results
print("Generating visualizations...")

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix - Email Spam Detection", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output_screenshot_confusion_matrix.png")
plt.close()

# 2. Top TF-IDF Features
indices = np.argsort(tfidf.idf_)[::-1]
features = tfidf.get_feature_names_out()
top_n = 20
plt.figure(figsize=(10, 8))
plt.barh(features[indices[:top_n]], tfidf.idf_[indices[:top_n]], color='salmon')
plt.title(f"Top {top_n} Important Words (TF-IDF)", fontsize=16)
plt.xlabel("IDF Score")
plt.tight_layout()
plt.savefig("output_screenshot_top_words.png")
plt.close()

# 3. Class Distribution
plt.figure(figsize=(8, 5))
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#99ff99'])
plt.title("Spam vs Ham - Data Distribution", fontsize=16)
plt.ylabel("")
plt.tight_layout()
plt.savefig("output_screenshot_distribution.png")
plt.close()

# Save results to a Markdown File
with open("Model_Outputs_Record.md", "w", encoding="utf-8") as f:
    f.write("# 📧 Email Spam Detection - Pro Results\n\n")
    f.write(f"**Accuracy:** {accuracy*100:.2f}%\n\n")
    f.write("### Classification Report\n")
    f.write("```text\n" + report + "```\n\n")
    f.write("## Graphical Analysis\n\n")
    f.write("### 1. Accuracy Heatmap (Confusion Matrix)\n")
    f.write("![Confusion Matrix](output_screenshot_confusion_matrix.png)\n\n")
    f.write("### 2. Dataset Balance\n")
    f.write("![Distribution](output_screenshot_distribution.png)\n\n")
    f.write("### 3. Key Keywords Analyzed\n")
    f.write("![Top Words](output_screenshot_top_words.png)\n")

# Update README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write("# 📧 Email Spam Detection System\n\n")
    f.write("A Natural Language Processing (NLP) tool to classify emails as Spam or Ham (Not Spam).\n\n")
    f.write("## 🛠 Features\n")
    f.write("- **TF-IDF Vectorization** for text processing.\n")
    f.write("- **Logistic Regression** for fast and reliable classification.\n")
    f.write("- Interactive **Streamlit App** to check custom emails.\n\n")
    f.write("## 📊 Stats\n")
    f.write(f"- Test Accuracy: `{accuracy*100:.2f}%`\n\n")
    f.write("## 💻 Usage\n")
    f.write("1. Train: `python train.py`\n")
    f.write("2. Run app: `streamlit run app.py`\n")

print("Project 2 records updated!")
