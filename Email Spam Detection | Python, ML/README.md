Here is the GitHub README documentation for the **Email Spam Detection** project, which is designed to detail the project's processes and results comprehensively, showcasing the work to potential recruiters.

---

# Email Spam Detection

## Introduction

This project focuses on detecting spam in email content using machine learning algorithms. With the ever-increasing volume of spam emails, a reliable and efficient classification system is crucial for maintaining email security and improving user experience. The project leverages natural language processing (NLP) techniques to preprocess and classify emails accurately.

## Project Overview

This project explores various machine learning models, including ensemble methods, to classify emails as spam or ham (not spam). We preprocess email content with NLP techniques, extract relevant features, and evaluate the performance of several classification models. This README serves to showcase the project's workflow, methodology, and results.

## Key Features

- **Data Preprocessing:** Tokenization, stemming, and removal of stop words to prepare text data.
- **Feature Engineering:** Added custom features like word count and sentence count for improved model accuracy.
- **Model Training and Evaluation:** Implemented multiple models with cross-validation to achieve high accuracy.
- **Ensemble Techniques:** Utilized voting classifiers and Stratified K-Fold cross-validation for enhanced precision.
- **Results Interpretation:** Detailed analysis of model results to determine the most effective approach.

## Dataset

The dataset used in this project is a labeled collection of spam and ham emails provided in `spam.csv`. It contains two main columns:
- `Category`: Identifies whether an email is spam or ham.
- `Message`: Contains the text content of the email.

## Project Workflow

### 1. Importing Required Libraries and Loading Data

We began by importing essential libraries for data processing, visualization, and modeling. The following Python libraries were used:

```python
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
```

After setting up the environment, we loaded and previewed the dataset:

```python
df = pd.read_csv("spam.csv")
df.head()
```

### 2. Data Cleaning

To ensure data integrity, we:
- Checked for null values and duplicates.
- Removed duplicates to avoid biased results.

```python
df.info()
df.isna().sum()
df.duplicated().sum()
df = df.drop_duplicates(keep="first")
```

### 3. Exploratory Data Analysis

Key exploratory steps included:
- **Encoding Categories:** Transformed the `Category` column into numerical values for modeling purposes.
- **Feature Engineering:** Added custom features such as `number_of_characters`, `number_of_words`, and `number_of_sentences`.
- **Statistical Analysis:** Explored basic statistics for spam and ham emails, highlighting differences in length.

```python
encoder = LabelEncoder()
df["Category"] = encoder.fit_transform(df["Category"])
df["number_of_characters"] = df["Message"].apply(len)
df["number_of_words"] = df["Message"].apply(lambda x: len(nltk.word_tokenize(x)))
df["number_of_sentences"] = df["Message"].apply(lambda x: len(nltk.sent_tokenize(x)))
```

### 4. Text Preprocessing

We defined a function to preprocess the email content by:
- Converting text to lowercase.
- Removing stop words and punctuation.
- Applying stemming for word simplification.

```python
ps = PorterStemmer()
def transform_text(text):
    text_lowercase = text.lower()
    word_tokens = nltk.word_tokenize(text_lowercase)
    word_tokens = [word for word in word_tokens if word.isalnum()]
    cleaned_text = [word for word in word_tokens if word not in stopwords.words('english') and word not in string.punctuation]
    stemmed_text = [ps.stem(word) for word in cleaned_text]
    return " ".join(stemmed_text)

df["Processed_Text"] = df["Message"].apply(transform_text)
```

### 5. Data Visualization

Created visual representations to better understand the data:
- **Word Clouds:** Visualized common words in spam and ham messages.
- **Bar Plots:** Displayed the top 20 words in spam messages.

```python
from wordcloud import WordCloud
# Generate word cloud for spam messages
spam_text = df[df["Category"] == 1]["Processed_Text"].str.cat(sep=" ")
spam_wordcloud = WordCloud(width=500, height=500, min_font_size=10, background_color="white").generate(spam_text)
plt.imshow(spam_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

### 6. Model Building

Several machine learning models were trained, including:
- **Logistic Regression**
- **Support Vector Classifier**
- **Naive Bayes Variants**
- **Decision Tree**
- **Ensemble Models** (Random Forest, Extra Trees, AdaBoost, Gradient Boosting)

We split the data into training and testing sets and applied vectorization using `CountVectorizer` and `TfidfVectorizer` for text representation.

```python
X_train, X_test, y_train, y_test = train_test_split(df.Processed_Text, df.Category, test_size=0.2, random_state=5)
cv = CountVectorizer()
tfidf = TfidfVectorizer()
X_train_count = cv.fit_transform(X_train.values).toarray()
X_test_count = cv.transform(X_test.values).toarray()
```

### 7. Model Evaluation

We evaluated each model's accuracy, precision, and recall, highlighting the top-performing models:
1. **Multinomial Naive Bayes (CountVectorizer)**
2. **Support Vector Classifier (TfidfVectorizer)**
3. **Extra Trees Classifier (TfidfVectorizer)**

These models were chosen based on their high accuracy and spam identification precision.

### 8. Model Improvement

To enhance the model further, we:
- Introduced `max_features` limit in `TfidfVectorizer` to focus on the most important terms.
- Utilized a **Voting Classifier** with a combination of the top 3 models to achieve a slight improvement in spam precision.
- Applied **Stratified K-Fold Cross-Validation** to achieve stable performance scores across different data splits.

### Results or Outcome

The final model results indicate that our approach achieved optimal results using the Voting Classifier and Stratified K-Fold Cross-Validation. Our top-performing classifiers successfully identify spam emails with high precision and recall.

### Conclusion

The Email Spam Detection project demonstrates a structured approach to text classification with high-performing models. Although the current implementation provides reliable results, future work could explore hyperparameter tuning with Stratified K-Fold Cross-Validation to further enhance the model, recognizing that this would require substantial computational resources.

---

This documentation outlines each step in the project, providing clear insights into the modeling decisions and outcomes. It demonstrates a thorough understanding of data processing, model selection, and evaluation methods, effectively showcasing the project for potential recruiters.
