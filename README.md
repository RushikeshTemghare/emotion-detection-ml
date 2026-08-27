# Emotion Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat\&logo=python\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat\&logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat\&logo=matplotlib\&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat\&logo=scikit-learn\&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3C873A?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat\&logo=jupyter\&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-blue?style=flat)
![Classification](https://img.shields.io/badge/Classification-Multiclass-red?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

## 📌 Project Overview

This project implements an end-to-end **emotion detection system using machine learning and natural language processing (NLP)**.

The objective is to classify text into different emotion categories by preprocessing textual data, converting the text into numerical features, training multiple machine learning models, and comparing their classification performance.

The project explores two different text feature extraction techniques:

* **Bag-of-Words (BoW)**
* **Term Frequency–Inverse Document Frequency (TF-IDF)**

These representations are used with different classification algorithms to evaluate their effectiveness for emotion classification.

---

## 🎯 Project Objective

The main objectives of this project are to:

* Load and inspect a labelled emotion dataset.
* Convert categorical emotion labels into numerical values.
* Clean and preprocess textual data.
* Remove punctuation, numbers, non-ASCII characters and stopwords.
* Split the dataset into training and testing sets.
* Convert text into numerical features using Bag-of-Words and TF-IDF.
* Train multiple machine learning classification models.
* Evaluate model performance using accuracy.
* Compare the performance of the different approaches.

---

## 📊 Dataset

The project uses a labelled text dataset stored in `train.txt`.

Each record contains two fields:

| Column    | Description                         |
| --------- | ----------------------------------- |
| `text`    | Text sentence expressing an emotion |
| `emotion` | Corresponding emotion category      |

The dataset is loaded using Pandas with a semicolon (`;`) as the delimiter.

```python
df = pd.read_csv(
    'train.txt',
    sep=';',
    header=None,
    names=['text', 'emotion']
)
```

Before model training, the emotion categories are converted into numerical labels so they can be used as the target variable for classification.

> **Dataset note:** If the dataset is subject to redistribution restrictions, `train.txt` should not be uploaded to the repository. In that case, provide the appropriate dataset source and instructions for placing the file in the project directory.

---

## 🔄 Project Workflow

The project follows the following machine learning pipeline:

```text
Raw Dataset
     ↓
Data Loading
     ↓
Dataset Inspection
     ↓
Emotion Label Encoding
     ↓
Text Preprocessing
     ├── Lowercase Conversion
     ├── Punctuation Removal
     ├── Number Removal
     ├── Non-ASCII Character Removal
     └── Stopword Removal
     ↓
Train/Test Split
     ↓
Text Feature Extraction
     ├── Bag-of-Words
     └── TF-IDF
     ↓
Machine Learning Models
     ├── Multinomial Naive Bayes
     └── Logistic Regression
     ↓
Emotion Prediction
     ↓
Accuracy Evaluation
     ↓
Model Comparison
```

---

## 🧹 Text Preprocessing

Text preprocessing is performed to clean and standardise the input data before machine learning.

The following steps are implemented:

### 1. Lowercase Conversion

All text is converted to lowercase so that words with different capitalisation are treated consistently.

### 2. Punctuation Removal

Punctuation characters are removed using Python's `string` module.

### 3. Number Removal

Numerical characters are removed from the text to reduce unnecessary features.

### 4. Non-ASCII Character Removal

Non-ASCII characters, including emojis and other characters outside the ASCII character set, are removed.

### 5. Stopword Removal

Common English stopwords are removed using the NLTK stopword corpus.

The result is a cleaner text representation that can be converted into numerical features for machine learning.

---

## 🔢 Feature Extraction

Machine learning algorithms cannot directly process raw text. Therefore, the cleaned text is converted into numerical representations.

### Bag-of-Words

The first approach uses `CountVectorizer` to implement a Bag-of-Words representation.

Bag-of-Words represents each document based on the frequency of words appearing in the text.

```python
bow_vectorizer = CountVectorizer()

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
```

### TF-IDF

The second approach uses `TfidfVectorizer`.

TF-IDF represents text by assigning weights to words based on their importance within a document relative to the overall collection of documents.

```python
tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

---

## 🤖 Machine Learning Models

Three supervised machine learning approaches are implemented.

### 1. Multinomial Naive Bayes + Bag-of-Words

The first model uses Bag-of-Words features with a Multinomial Naive Bayes classifier.

```python
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
```

The trained model is then used to predict the emotion labels of the testing data.

---

### 2. Multinomial Naive Bayes + TF-IDF

The second approach uses TF-IDF features with another Multinomial Naive Bayes classifier.

```python
nb2_model = MultinomialNB()
nb2_model.fit(X_train_tfidf, y_train)
```

This allows the performance of Naive Bayes using TF-IDF to be compared against the Bag-of-Words approach.

---

### 3. Logistic Regression + TF-IDF

The third approach uses the TF-IDF representation with Logistic Regression.

```python
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)
```

The model learns the relationship between the TF-IDF features and the encoded emotion categories.

---

## 📈 Model Evaluation

The models are evaluated using **classification accuracy**.

Accuracy is calculated by comparing the predicted emotion labels against the actual labels in the testing dataset.

The project stores the three accuracy scores as:

```python
bow_accuracy
tfidf_nb_accuracy
logistic_accuracy
```

This allows the performance of the three approaches to be compared directly.

### Model Comparison

| Model                   | Feature Representation | Evaluation Metric |
| ----------------------- | ---------------------- | ----------------- |
| Multinomial Naive Bayes | Bag-of-Words           | Accuracy          |
| Multinomial Naive Bayes | TF-IDF                 | Accuracy          |
| Logistic Regression     | TF-IDF                 | Accuracy          |

The final accuracy values can be viewed directly in the Jupyter Notebook.

---

## 🛠️ Technologies & Libraries

* **Python** – Core programming language
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualisation
* **Seaborn** – Data visualisation
* **NLTK** – Natural language processing and stopword handling
* **Scikit-learn** – Feature extraction, machine learning models and evaluation
* **Jupyter Notebook** – Development and documentation environment

---

## 📂 Project Structure

```text
emotion-detection-ml/
│
├── emotion_detection.ipynb
├── train.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/RushikeshTemghare/emotion-detection-ml.git
```

### 2. Navigate to the project directory

```bash
cd emotion-detection-ml
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
emotion_detection.ipynb
```

### 5. Run the notebook

Run the cells sequentially from the beginning of the notebook.

The notebook will:

* Load the dataset
* Preprocess the text
* Encode the emotion labels
* Create Bag-of-Words and TF-IDF features
* Train the three machine learning models
* Generate predictions
* Calculate model accuracy
* Compare the results

---

## 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Text preprocessing
* Natural Language Processing
* Stopword removal
* Categorical label encoding
* Train/test splitting
* Bag-of-Words representation
* TF-IDF feature extraction
* Multinomial Naive Bayes
* Logistic Regression
* Supervised machine learning
* Text classification
* Model evaluation using accuracy
* Comparing machine learning approaches

---

## 👤 Author

**Rushikesh Temghare**

MSc Data Science & Artificial Intelligence
Bournemouth University

Passionate about Machine Learning, Artificial Intelligence, Data Analytics, and Data Science.

[![GitHub](https://img.shields.io/badge/GitHub-RushikeshTemghare-181717?style=flat\&logo=github\&logoColor=white)](https://github.com/RushikeshTemghare)

---

## 📄 License

This project is licensed under the MIT License.
