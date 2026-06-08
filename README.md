# 📩 SMS Spam Classifier using Machine Learning

## 📌 Project Overview
This project is a Machine Learning-based SMS Spam Detection system that classifies messages as Spam or Legitimate (Ham) using Natural Language Processing (NLP) techniques.

It uses TF-IDF vectorization to convert text into numerical form and Logistic Regression for classification.

---

## 🚀 Features
- Classifies SMS messages as Spam or Legitimate
- Uses TF-IDF for text feature extraction
- Machine Learning model (Logistic Regression)
- Real-time user input prediction
- High accuracy (~95%)

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer (NLP)

---

## 📂 Dataset
- SMS Spam Collection Dataset (`spam.csv`)
- Labels:
  - ham → Legitimate message
  - spam → Spam message

---

## ⚙️ How It Works
1. Load dataset
2. Clean and preprocess text data
3. Convert text into numerical features using TF-IDF
4. Train Logistic Regression model
5. Evaluate accuracy
6. Predict new SMS messages

---

## ▶️ How to Run

### Install dependencies
```bash
pip install pandas scikit-learn numpy
