# coding: utf-8
"""
Train NLP Document Classification models.
Techniques: TF-IDF, Bag-of-Words, Preprocessing (NLTK)
Classifiers: Naive Bayes, Logistic Regression, SVM, Random Forest, Decision Tree, KNN
Author : Shrawani Joshi
Project: ML Minor Project - Document Classification using NLP
"""

import os, pickle, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
nltk.download('stopwords',    quiet=True)
nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('wordnet',      quiet=True)
nltk.download('omw-1.4',      quiet=True)

from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing    import LabelEncoder
from sklearn.naive_bayes      import MultinomialNB, ComplementNB
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (accuracy_score, f1_score, classification_report,
                                      confusion_matrix)

warnings.filterwarnings('ignore')

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, 'data', 'documents.csv')
MODEL = os.path.join(BASE, 'models')
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(MODEL, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print('=' * 68)
print('  ML MINOR PROJECT - DOCUMENT CLASSIFICATION USING NLP')
print('  Author : Shrawani Joshi')
print('=' * 68)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f'\nDataset Shape : {df.shape}')
print('Category Distribution:')
print(df['category'].value_counts())

CATEGORIES = sorted(df['category'].unique())
COLORS     = ['#E53935','#8E24AA','#1E88E5','#43A047','#FB8C00','#00ACC1','#6D4C41']
cat_color  = dict(zip(CATEGORIES, COLORS))

# ══════════════════════════════════════════════════════════════════════════════
# PART A: TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('PART A: NLP Text Preprocessing')
print('='*68)

STOP_WORDS  = set(stopwords.words('english'))
stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()

def preprocess(text, use_lemma=True):
    text  = text.lower()
    text  = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens= word_tokenize(text)
    tokens= [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    if use_lemma:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

print('Preprocessing documents...')
df['text_clean']  = df['text'].apply(preprocess)
df['text_stem']   = df['text'].apply(lambda x: preprocess(x, use_lemma=False))
df['word_count']  = df['text'].apply(lambda x: len(x.split()))
df['clean_count'] = df['text_clean'].apply(lambda x: len(x.split()))
print(f'Avg words before cleaning : {df["word_count"].mean():.1f}')
print(f'Avg words after cleaning  : {df["clean_count"].mean():.1f}')
print('Sample preprocessed text:')
print(df[['category','text_clean']].head(3).to_string())

# ── Encode Labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])

X_raw   = df['text']
X_clean = df['text_clean']
y       = df['label']

X_tr_r,  X_te_r,  y_tr, y_te = train_test_split(X_raw,   y, test_size=0.2, random_state=42, stratify=y)
X_tr_c,  X_te_c,  _,    _    = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)

# ══════════════════════════════════════════════════════════════════════════════
# PART B: FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('PART B: Feature Extraction - TF-IDF & Bag of Words')
print('='*68)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_tfidf_tr = tfidf.fit_transform(X_tr_c)
X_tfidf_te = tfidf.transform(X_te_c)
print(f'TF-IDF feature matrix : {X_tfidf_tr.shape}')

# Bag of Words
bow = CountVectorizer(max_features=5000, ngram_range=(1,1), min_df=2)
X_bow_tr = bow.fit_transform(X_tr_c)
X_bow_te = bow.transform(X_te_c)
print(f'BoW feature matrix    : {X_bow_tr.shape}')

# ══════════════════════════════════════════════════════════════════════════════
# PART C: MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('PART C: Training Classifiers')
print('='*68)

classifiers = {
    'Multinomial Naive Bayes':   MultinomialNB(alpha=0.1),
    'Complement Naive Bayes':    ComplementNB(alpha=0.1),
    'Logistic Regression':       LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Linear SVM':                LinearSVC(C=1.0, max_iter=2000, random_state=42),
    'Random Forest':             RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree':             DecisionTreeClassifier(max_depth=10, random_state=42),
    'K-Nearest Neighbors':       KNeighborsClassifier(n_neighbors=5),
}

results_tfidf = []
results_bow   = []
best_name, best_model, best_vec, best_acc = '', None, None, 0

for name, clf in classifiers.items():
    # TF-IDF
    clf.fit(X_tfidf_tr, y_tr)
    if hasattr(clf, 'predict_proba'):
        preds = clf.predict(X_tfidf_te)
    else:
        preds = clf.predict(X_tfidf_te)
    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average='weighted')
    results_tfidf.append({'Model': name, 'Accuracy(%)': round(acc*100,2), 'F1(weighted)': round(f1,4)})
    print(f'\n[TF-IDF] {name}\n  Accuracy={acc*100:.2f}%  F1={f1:.4f}')
    if acc > best_acc:
        best_acc, best_name, best_model, best_vec = acc, name + ' (TF-IDF)', clf, tfidf

    # BoW (skip LinearSVC for BoW to avoid issues)
    try:
        clf2 = type(clf)(**clf.get_params())
        clf2.fit(X_bow_tr, y_tr)
        preds2 = clf2.predict(X_bow_te)
        acc2   = accuracy_score(y_te, preds2)
        f1_2   = f1_score(y_te, preds2, average='weighted')
        results_bow.append({'Model': name, 'Accuracy(%)': round(acc2*100,2), 'F1(weighted)': round(f1_2,4)})
        print(f'[BoW   ] {name}\n  Accuracy={acc2*100:.2f}%  F1={f1_2:.4f}')
    except Exception:
        results_bow.append({'Model': name, 'Accuracy(%)': 0, 'F1(weighted)': 0})

tfidf_df = pd.DataFrame(results_tfidf).sort_values('Accuracy(%)', ascending=False)
bow_df   = pd.DataFrame(results_bow).sort_values('Accuracy(%)', ascending=False)

print('\n--- TF-IDF Results ---')
print(tfidf_df.to_string(index=False))
print('\n--- BoW Results ---')
print(bow_df.to_string(index=False))
print(f'\nBest Model: {best_name}  (Accuracy={best_acc*100:.2f}%)')
print('\nDetailed Classification Report (Best Model):')
print(classification_report(y_te, best_model.predict(X_tfidf_te), target_names=le.classes_))

# Cross-validation on best
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_tfidf_tr, y_tr, cv=cv, scoring='accuracy')
print(f'\n5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)')

# ── Save Models ────────────────────────────────────────────────────────────────
pickle.dump(best_model,  open(os.path.join(MODEL, 'classifier.pkl'),  'wb'))
pickle.dump(tfidf,       open(os.path.join(MODEL, 'tfidf.pkl'),       'wb'))
pickle.dump(bow,         open(os.path.join(MODEL, 'bow.pkl'),         'wb'))
pickle.dump(le,          open(os.path.join(MODEL, 'label_encoder.pkl'),'wb'))
pickle.dump(tfidf_df,    open(os.path.join(MODEL, 'tfidf_results.pkl'),'wb'))
pickle.dump(bow_df,      open(os.path.join(MODEL, 'bow_results.pkl'), 'wb'))
print('\nModels saved.')

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Category Distribution
fig, ax = plt.subplots(figsize=(10, 5))
counts = df['category'].value_counts().reindex(CATEGORIES)
bars = ax.bar(CATEGORIES, counts.values, color=COLORS, edgecolor='white', width=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
            str(val), ha='center', fontweight='bold', fontsize=10)
ax.set_title('Document Category Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Documents')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'01_category_distribution.png'), dpi=150)
plt.close()

# 2. Word Count Distribution
fig, ax = plt.subplots(figsize=(10, 5))
for i, cat in enumerate(CATEGORIES):
    sns.kdeplot(df[df['category']==cat]['word_count'], ax=ax,
                color=COLORS[i], fill=True, alpha=0.3, label=cat)
ax.set_title('Word Count Distribution by Category', fontsize=14, fontweight='bold')
ax.set_xlabel('Word Count'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'02_word_count_distribution.png'), dpi=150)
plt.close()

# 3. Word Clouds per category
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, cat in enumerate(CATEGORIES):
    texts = ' '.join(df[df['category']==cat]['text_clean'])
    wc = WordCloud(width=400, height=250, background_color='white',
                   colormap='tab10', max_words=60, random_state=42).generate(texts)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    axes[i].set_title(cat, fontsize=12, fontweight='bold', color=COLORS[i])
axes[-1].axis('off')
fig.suptitle('Word Clouds by Category (After Preprocessing)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'03_word_clouds.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4. TF-IDF Model Comparison
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(tfidf_df['Model'], tfidf_df['Accuracy(%)'],
               color='#1E88E5', edgecolor='white')
ax.set_xlim(0, 115)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison (TF-IDF)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, tfidf_df['Accuracy(%)']):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
            f'{val}%', va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'04_tfidf_model_comparison.png'), dpi=150)
plt.close()

# 5. BoW Model Comparison
valid_bow = bow_df[bow_df['Accuracy(%)'] > 0]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(valid_bow['Model'], valid_bow['Accuracy(%)'],
               color='#43A047', edgecolor='white')
ax.set_xlim(0, 115)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison (Bag of Words)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, valid_bow['Accuracy(%)']):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
            f'{val}%', va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'05_bow_model_comparison.png'), dpi=150)
plt.close()

# 6. Confusion Matrix (best model)
cm = confusion_matrix(y_te, best_model.predict(X_tfidf_te))
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=13, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'06_confusion_matrix.png'), dpi=150)
plt.close()

# 7. TF-IDF vs BoW Comparison
common = set(tfidf_df['Model']) & set(valid_bow['Model'])
tfidf_c = tfidf_df[tfidf_df['Model'].isin(common)].set_index('Model')['Accuracy(%)']
bow_c   = valid_bow[valid_bow['Model'].isin(common)].set_index('Model')['Accuracy(%)']
compare = pd.DataFrame({'TF-IDF': tfidf_c, 'BoW': bow_c}).dropna()
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(compare)); w = 0.35
ax.bar(x - w/2, compare['TF-IDF'], w, label='TF-IDF', color='#1E88E5')
ax.bar(x + w/2, compare['BoW'],    w, label='Bag of Words', color='#43A047')
ax.set_xticks(x); ax.set_xticklabels(compare.index, rotation=20, ha='right')
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 115)
ax.set_title('TF-IDF vs Bag of Words - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'07_tfidf_vs_bow.png'), dpi=150)
plt.close()

# 8. Top TF-IDF Features per category
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
feature_names = np.array(tfidf.get_feature_names_out())
for i, cat in enumerate(CATEGORIES):
    cat_idx = le.transform([cat])[0]
    cat_mask = (y == cat_idx)
    cat_texts = X_clean[cat_mask]
    vect_temp = TfidfVectorizer(max_features=3000, ngram_range=(1,1), min_df=1)
    mat = vect_temp.fit_transform(cat_texts)
    scores = mat.toarray().mean(axis=0)
    top_idx = scores.argsort()[-15:][::-1]
    top_words  = np.array(vect_temp.get_feature_names_out())[top_idx]
    top_scores = scores[top_idx]
    axes[i].barh(top_words[::-1], top_scores[::-1], color=COLORS[i])
    axes[i].set_title(cat, fontsize=11, fontweight='bold', color=COLORS[i])
    axes[i].set_xlabel('Avg TF-IDF Score', fontsize=8)
axes[-1].axis('off')
fig.suptitle('Top 15 TF-IDF Keywords per Category', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'08_top_keywords.png'), dpi=150)
plt.close()

# 9. F1 Score Comparison
fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(tfidf_df['Model'], tfidf_df['F1(weighted)'],
        color='#FB8C00', edgecolor='white')
ax.set_xlim(0, 1.1)
ax.set_xlabel('F1 Score (Weighted)', fontsize=12)
ax.set_title('F1 Score Comparison (TF-IDF)', fontsize=14, fontweight='bold')
for i, (_, row) in enumerate(tfidf_df.iterrows()):
    ax.text(row['F1(weighted)']+0.01, i, f'{row["F1(weighted)"]:.4f}',
            va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'09_f1_comparison.png'), dpi=150)
plt.close()

# 10. Before vs After Preprocessing Word Count
fig, ax = plt.subplots(figsize=(10, 5))
cats_list = list(CATEGORIES)
before = [df[df['category']==c]['word_count'].mean()  for c in cats_list]
after  = [df[df['category']==c]['clean_count'].mean() for c in cats_list]
x = np.arange(len(cats_list)); w = 0.35
ax.bar(x-w/2, before, w, label='Before Preprocessing', color='#E53935', alpha=0.8)
ax.bar(x+w/2, after,  w, label='After Preprocessing',  color='#43A047', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(cats_list, rotation=20, ha='right')
ax.set_ylabel('Average Word Count'); ax.legend(fontsize=11)
ax.set_title('Word Count Before vs After Preprocessing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,'10_preprocessing_effect.png'), dpi=150)
plt.close()

print('\nAll 10 plots saved to /plots/')
print('\nTraining complete!')
print(f'Best Model: {best_name}  Accuracy={best_acc*100:.2f}%')
