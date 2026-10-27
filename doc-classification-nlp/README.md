# Document Classification using NLP

**ML Major Project**  
**Author:** Shrawani Shrikant Joshi  
**Subject:** Machine Learning  
**Year:** 2026-27  

---

## Project Overview

This project implements a **Document Classification System** using Natural Language Processing (NLP) techniques.
The system classifies text documents into predefined categories using ML algorithms trained on TF-IDF and Bag-of-Words features.

The app also supports **custom dataset upload** — upload any CSV or Excel file and train models on your own data in real time.

---

## Categories (Default Dataset)

| Category | Description |
|----------|-------------|
| Technology | AI, software, hardware, cloud, programming |
| Sports | Football, cricket, tennis, Olympics, tournaments |
| Politics | Government, elections, parliament, diplomacy |
| Entertainment | Movies, music, celebrities, streaming, awards |
| Business | Stock market, companies, economy, finance |
| Health | Medicine, vaccines, hospitals, clinical trials |
| Science | Space, research, physics, chemistry, NASA |

---

## NLP Techniques Used

| Technique | Tool | Purpose |
|-----------|------|---------|
| Tokenization | NLTK | Split text into individual words |
| Stop Word Removal | NLTK Stopwords | Remove common words (the, is, a...) |
| Lemmatization | WordNet (NLTK) | Reduce words to their base form |
| TF-IDF Vectorization | scikit-learn | Convert text to numerical features |
| Bag of Words (BoW) | scikit-learn | Word frequency based features |
| Word Cloud | WordCloud library | Visual keyword representation |

---

## Machine Learning Models

| Model | Accuracy (TF-IDF) |
|-------|-------------------|
| Multinomial Naive Bayes | 100.00% |
| Complement Naive Bayes | 100.00% |
| Logistic Regression | 100.00% |
| Linear SVM | 100.00% |
| Random Forest | 100.00% |
| K-Nearest Neighbors | 100.00% |
| Decision Tree | 95.24% |

**Best Model:** Multinomial Naive Bayes (TF-IDF)  
**5-Fold Cross Validation Accuracy:** 100.00% +/- 0.00%

---

## Project Structure

```
doc-classification-nlp/
|
|-- generate_dataset.py       # Generate synthetic training dataset
|-- train_model.py            # Train NLP models and generate plots
|-- app.py                    # Streamlit web application
|-- requirements.txt          # Python dependencies
|-- README.md                 # Project documentation
|-- .gitignore
|
|-- data/
|   |-- documents.csv         # Generated dataset (2100 documents)
|
|-- models/
|   |-- classifier.pkl        # Trained best model
|   |-- tfidf.pkl             # TF-IDF vectorizer
|   |-- bow.pkl               # Bag of Words vectorizer
|   |-- label_encoder.pkl     # Label encoder
|   |-- tfidf_results.pkl     # Results dataframe
|   |-- bow_results.pkl       # BoW results dataframe
|
|-- plots/
    |-- 01_category_distribution.png
    |-- 02_word_count_distribution.png
    |-- 03_word_clouds.png
    |-- 04_tfidf_model_comparison.png
    |-- 05_bow_model_comparison.png
    |-- 06_confusion_matrix.png
    |-- 07_tfidf_vs_bow.png
    |-- 08_top_keywords.png
    |-- 09_f1_comparison.png
    |-- 10_preprocessing_effect.png
```

---

## Installation & Setup

### Step 1 - Install Python
Make sure Python 3.8 or higher is installed on your system.

### Step 2 - Install Required Libraries

```bash
pip install numpy pandas scikit-learn streamlit matplotlib seaborn nltk wordcloud openpyxl
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1 - Generate Dataset

```bash
python generate_dataset.py
```

Creates `data/documents.csv` with 2100 synthetic news documents (300 per category).

### Step 2 - Train the Models

```bash
python train_model.py
```

This will:
- Preprocess all text (tokenize, remove stop words, lemmatize)
- Extract TF-IDF and Bag of Words features
- Train 7 machine learning models and compare them
- Save 10 analysis plots to `plots/` folder
- Save trained models to `models/` folder

### Step 3 - Launch the Web App

```bash
streamlit run app.py
```

Open browser at: **http://localhost:8501**

---

## Web App Features

### Tab 1 - Classify Document
- Type or paste any document text
- Choose from sample texts for each category
- Get predicted category with confidence score
- Interactive probability bar chart for all categories
- Auto-generated word cloud from your input text

### Tab 2 - Dataset Analysis
- Category distribution chart and pie chart
- Word count distribution per category
- Word clouds for all 7 categories
- Top TF-IDF keywords per category
- Before vs After preprocessing word count comparison

### Tab 3 - NLP Pipeline
- Step-by-step explanation of the preprocessing pipeline
- TF-IDF formula and explanation
- Bag of Words vs TF-IDF comparison
- Libraries and tools used

### Tab 4 - Model Results
- All 7 model accuracy comparison
- F1 Score comparison chart
- Confusion matrix heatmap
- TF-IDF vs Bag of Words performance bar chart

---

## Upload Your Own Dataset

The app supports **custom dataset upload** feature!

### Steps:
1. In the sidebar, select **'Upload My Own Dataset'**
2. Click **Browse files** and upload your `.csv` or `.xlsx` file
3. Select the **Text Column** (column containing document text)
4. Select the **Label Column** (column containing category names)
5. Choose feature extraction: **TF-IDF** or **Bag of Words**
6. App automatically trains **6 ML models** on your data
7. Type any document to classify it using your trained models

### Expected CSV Format:

```csv
text,category
"Artificial intelligence is transforming technology",Technology
"The football team won the championship match",Sports
"Parliament passed a new budget reform bill",Politics
```

**Supported formats:** `.csv` | `.xlsx` | `.xls`

---

## Text Preprocessing Pipeline

```
Raw Text
   |
   v
Lowercase Conversion
   |
   v
Remove Punctuation and Numbers (regex)
   |
   v
Tokenization (NLTK word_tokenize)
   |
   v
Stop Word Removal (NLTK stopwords - 179 words removed)
   |
   v
Lemmatization (WordNet Lemmatizer)
   |
   v
TF-IDF Vectorization (max 5000 features, unigrams + bigrams)
   |
   v
ML Classifier -> Category Prediction
```

---

## Key Concepts

### TF-IDF (Term Frequency - Inverse Document Frequency)

```
TF-IDF(t, d) = TF(t, d) * log(N / df(t))

TF  = (Number of times term t appears in document d) / (Total terms in d)
IDF = log(Total documents N / Documents containing term t)
```

- Common words like 'the', 'is' get very low TF-IDF scores
- Rare but meaningful words get high TF-IDF scores
- Better than simple word counts for document classification

### Why Multinomial Naive Bayes for Text?
- Based on Bayes theorem: P(category | words) proportional to P(words | category) * P(category)
- Works extremely well with TF-IDF and word frequency features
- Fast to train, even on large datasets
- Low memory usage

### Lemmatization vs Stemming

| Approach | Example | Result |
|----------|---------|--------|
| Lemmatization | running, runs, ran | run (correct dictionary word) |
| Stemming | running, runs | run (rule-based, may not be a real word) |

This project uses **Lemmatization** (WordNet) for better accuracy.

---

## Requirements

```
numpy
pandas
scikit-learn
streamlit
matplotlib
seaborn
nltk
wordcloud
openpyxl
```

---

## GitHub Upload Instructions

```bash
git init
git add .
git commit -m "Document Classification using NLP - ML Minor Project"
git remote add origin https://github.com/YOUR_USERNAME/doc-classification-nlp.git
git push -u origin main
```

> Note: Replace `YOUR_USERNAME` with your actual GitHub username.

---

## References

1. NLTK Documentation - https://www.nltk.org/
2. scikit-learn Text Feature Extraction - https://scikit-learn.org/stable/modules/feature_extraction.html
3. TF-IDF - https://en.wikipedia.org/wiki/Tf-idf
4. Multinomial Naive Bayes - https://scikit-learn.org/stable/modules/naive_bayes.html
5. Streamlit Docs - https://docs.streamlit.io/

---

*Document Classification using NLP | ML Minor Project | Shrawani Joshi | 2025-26*
