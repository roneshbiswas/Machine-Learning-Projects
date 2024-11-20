# Sentiment Analysis of Sales Agents' Interaction

This project aims to analyze sales agents' interactions with customers, using sentiment analysis techniques to evaluate agent performance and extract actionable insights. The project employs Python and pre-trained sentiment analysis models like VADER and RoBERTa.

## Dataset Overview

The dataset contains logs of interactions between sales agents and customers, including interaction IDs, email addresses, dates, types (Call/Email/SMS), and text transcripts. Example fields include:

- `interactionID`
- `fromEmailId`
- `toEmailId`
- `InteractionDate`
- `InteractionType(Call/Email/SMS)`
- `Extracted Interaction Text`

---

## Steps Involved

### 1. Data Cleaning
- Fixed column naming conventions.
- Removed null values and duplicates.

### 2. EDA (Exploratory Data Analysis) Phase 1
- Checked descriptive statistics of numerical columns.
- Reviewed value counts and proportions of categorical columns.
- Split the necessary columns for analysis.
- Examined interaction count and proportions for each sales agent.

### 3. EDA Phase 2
- Used NLTK packages to create new columns with:
  - Character count.
  - Word count.
  - Sentence count of interaction texts.
- Plotted pairplots and used different hue combinations for deeper insights.

### 4. EDA Phase 3
- Used NLTK packages to tokenize words:
  - Retained only alphanumeric words.
  - Removed stopwords.
- Visualized processed text using:
  - Word cloud.
  - Top 20 most frequently occurring words.

### 5. Generate Sentiment Scores
- Generated sentiment scores using:
  - **VADER**
  - **RoBERTa** pre-trained sentiment analysis models.
- Processed sentiment types for each model.

### 6. Models and Sales Agents Performance Evaluation
- Evaluated models by analyzing performance differences.
- Used the best-performing model to:
  - Analyze sales agents' performance.
  - Visualize word clouds for each sentiment type.
  - Visualize top 10 most frequently occurring words for each sentiment type.
  - Analyze character count, word count, and sentence count for each sentiment type.


