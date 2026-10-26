# coding: utf-8
import os, pickle, re, warnings, io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing           import LabelEncoder
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import LogisticRegression
from sklearn.svm                     import LinearSVC
from sklearn.ensemble                import RandomForestClassifier
from sklearn.tree                    import DecisionTreeClassifier
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.metrics                 import accuracy_score, f1_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Document Classifier - NLP',
    page_icon='D',
    layout='wide',
    initial_sidebar_state='expanded'
)

BASE  = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, 'models')
DATA  = os.path.join(BASE, 'data', 'documents.csv')
PLOTS = os.path.join(BASE, 'plots')

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ── NLP helpers ───────────────────────────────────────────────────────────────
def preprocess(text):
    text   = str(text).lower()
    text   = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

@st.cache_data(show_spinner='Preprocessing text...')
def preprocess_series(series):
    return series.apply(preprocess)

def train_on_df(df, text_col, label_col, vectorizer_type='tfidf'):
    df_clean            = df[[text_col, label_col]].dropna().copy()
    df_clean['clean']   = preprocess_series(df_clean[text_col])
    le                  = LabelEncoder()
    df_clean['label']   = le.fit_transform(df_clean[label_col].astype(str))
    X = df_clean['clean']
    y = df_clean['label']

    if vectorizer_type == 'tfidf':
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1)
    else:
        vec = CountVectorizer(max_features=5000, ngram_range=(1,1), min_df=1)

    X_vec   = vec.fit_transform(X)
    test_sz = min(0.2, max(0.05, 50 / len(df_clean)))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_vec, y, test_size=test_sz, random_state=42,
        stratify=y if len(le.classes_) > 1 else None
    )

    classifiers = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression':     LogisticRegression(max_iter=500, random_state=42),
        'Linear SVM':              LinearSVC(max_iter=2000, random_state=42),
        'Random Forest':           RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Decision Tree':           DecisionTreeClassifier(max_depth=10, random_state=42),
        'KNN':                     KNeighborsClassifier(n_neighbors=min(5, len(le.classes_))),
    }

    results      = []
    best_acc     = 0
    best_clf     = None
    best_clf_name= ''
    trained      = {}

    for name, clf in classifiers.items():
        try:
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            acc   = accuracy_score(y_te, preds)
            f1    = f1_score(y_te, preds, average='weighted', zero_division=0)
            results.append({'Model': name, 'Accuracy(%)': round(acc*100, 2), 'F1': round(f1, 4)})
            trained[name] = clf
            if acc > best_acc:
                best_acc, best_clf, best_clf_name = acc, clf, name
        except Exception as ex:
            results.append({'Model': name, 'Accuracy(%)': 0, 'F1': 0})

    return best_clf, vec, le, best_clf_name, best_acc, pd.DataFrame(results).sort_values('Accuracy(%)', ascending=False), df_clean, trained, X_te, y_te

# ── Load default models ───────────────────────────────────────────────────────
@st.cache_resource
def load_default_models():
    clf   = pickle.load(open(os.path.join(MODEL,'classifier.pkl'),   'rb'))
    tfidf = pickle.load(open(os.path.join(MODEL,'tfidf.pkl'),        'rb'))
    le    = pickle.load(open(os.path.join(MODEL,'label_encoder.pkl'),'rb'))
    tdf   = pickle.load(open(os.path.join(MODEL,'tfidf_results.pkl'),'rb'))
    return clf, tfidf, le, tdf

@st.cache_data
def load_default_data():
    return pd.read_csv(DATA)

try:
    def_clf, def_tfidf, def_le, def_tdf = load_default_models()
    def_df   = load_default_data()
    default_ready = True
except Exception as e:
    default_ready = False

# ── Header ────────────────────────────────────────────────────────────────────
HEADER = (
    '<div style="background:linear-gradient(135deg,#1a237e,#283593,#3949ab);'
    'padding:2.2rem 2rem;border-radius:16px;margin-bottom:1.5rem;text-align:center">'
    '<h1 style="color:white;font-size:2.4rem;margin:0">Document Classification using NLP</h1>'
    '<p style="color:#c5cae9;margin:0.5rem 0 0;font-size:1rem">'
    'TF-IDF + ML Classifiers | NLP Minor Project | '
    '<strong>Author: Shrawani Joshi</strong></p></div>'
)
st.markdown(HEADER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - DATASET SELECTION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('## Dataset Source')
    st.markdown('---')
    dataset_mode = st.radio(
        'Choose dataset:',
        ['Use Default Dataset', 'Upload My Own Dataset'],
        index=0
    )

    uploaded_file = None
    user_df       = None
    text_col      = None
    label_col     = None
    vec_type      = 'tfidf'

    if dataset_mode == 'Upload My Own Dataset':
        st.markdown('---')
        st.markdown('#### Upload Dataset')
        uploaded_file = st.file_uploader(
            'Upload CSV or Excel file',
            type=['csv', 'xlsx', 'xls'],
            help='Your file must have at least one text column and one label/category column.'
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)

                st.success(f'Loaded {len(user_df):,} rows x {len(user_df.columns)} columns')
                st.dataframe(user_df.head(3), use_container_width=True)

                all_cols = list(user_df.columns)
                st.markdown('#### Select Columns')
                text_col  = st.selectbox('Text Column (documents):', all_cols,
                                         index=0 if all_cols else 0)
                label_col = st.selectbox('Label Column (categories):', all_cols,
                                         index=min(1, len(all_cols)-1))
                vec_type  = st.selectbox('Feature Extraction:', ['tfidf', 'bow'],
                                         format_func=lambda x: 'TF-IDF' if x=='tfidf' else 'Bag of Words')

                if text_col == label_col:
                    st.error('Text and Label columns must be different!')
                    user_df = None
                else:
                    n_cats = user_df[label_col].nunique()
                    st.info(f'{n_cats} unique categories found: {", ".join(str(c) for c in user_df[label_col].unique()[:8])}{"..." if n_cats > 8 else ""}')

            except Exception as e:
                st.error(f'Error reading file: {e}')
                user_df = None

        st.markdown('---')
        st.markdown('#### Sample File Format')
        st.markdown('''
Your CSV should look like this:
```
text,category
"This is a sports article...",Sports
"Tech news about AI...",Technology
```
        ''')
        st.download_button(
            'Download Sample CSV',
            data='text,category\n"Artificial intelligence is transforming technology",Technology\n"The football match ended in victory",Sports\n"Parliament passed a new bill",Politics\n',
            file_name='sample_dataset.csv',
            mime='text/csv'
        )

# ══════════════════════════════════════════════════════════════════════════════
# Determine which model/data to use
# ══════════════════════════════════════════════════════════════════════════════
MAIN_TABS = st.tabs(['Classify Document', 'Dataset Analysis', 'NLP Pipeline', 'Model Results'])

# ── UPLOADED DATASET: TRAIN LIVE ─────────────────────────────────────────────
if dataset_mode == 'Upload My Own Dataset':
    if user_df is None or text_col is None:
        with MAIN_TABS[0]:
            st.info('Upload a CSV/Excel file in the sidebar to get started.')
            st.markdown('#### How to use:')
            st.markdown('''
1. In the **sidebar**, select **"Upload My Own Dataset"**
2. Click **Browse files** and upload your CSV or Excel file
3. Choose the **Text column** (the column containing document text)
4. Choose the **Label column** (the column containing category names)
5. The model will **train automatically** on your data
6. Then type any document in the text box to **classify it**
            ''')
        for tab in MAIN_TABS[1:]:
            with tab:
                st.info('Upload a dataset first using the sidebar.')
        st.stop()

    # Train
    with st.spinner(f'Training on your dataset ({len(user_df):,} documents)... Please wait.'):
        (active_clf, active_vec, active_le,
         best_clf_name, best_acc,
         results_df, df_clean,
         trained_clfs, X_te_mat, y_te_arr) = train_on_df(user_df, text_col, label_col, vec_type)

    st.success(f'Trained on your dataset! Best: **{best_clf_name}** — Accuracy: **{best_acc*100:.2f}%**')
    CATEGORIES   = list(active_le.classes_)
    active_df    = df_clean
    active_tdf   = results_df

else:
    # Default
    if not default_ready:
        for tab in MAIN_TABS:
            with tab:
                st.error('Default models not found. Run: python generate_dataset.py  then  python train_model.py')
        st.stop()
    active_clf  = def_clf
    active_vec  = def_tfidf
    active_le   = def_le
    active_tdf  = def_tdf
    CATEGORIES  = list(active_le.classes_)
    active_df   = def_df
    results_df  = def_tdf
    best_clf_name = type(active_clf).__name__
    best_acc    = 1.0
    X_te_mat    = None
    y_te_arr    = None

COLORS_LIST = ['#E53935','#8E24AA','#1E88E5','#43A047','#FB8C00','#00ACC1',
               '#6D4C41','#00897B','#F4511E','#546E7A','#7CB342','#039BE5']
cat_colors  = {cat: COLORS_LIST[i % len(COLORS_LIST)] for i, cat in enumerate(CATEGORIES)}

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with MAIN_TABS[0]:
    st.markdown('### Enter any document text to classify:')

    if dataset_mode == 'Use Default Dataset':
        samples = {
            'Technology': 'Artificial intelligence and machine learning are transforming software development. Deep neural networks and GPU computing enable faster model training on cloud platforms.',
            'Sports':     'The football championship final ended in a dramatic penalty shootout. The winning team celebrated their victory at the stadium after an outstanding performance.',
            'Politics':   'Parliament passed a new bill reforming the tax policy. The president signed the legislation after months of debate between the government and opposition.',
            'Health':     'Clinical trials for the new cancer vaccine showed promising results. The pharmaceutical company announced significant improvement in patient outcomes.',
            'Science':    'NASA scientists discovered a new exoplanet orbiting a distant star. Researchers published their findings about gravitational waves in a peer-reviewed journal.',
            'Business':   'The company reported record quarterly earnings driven by strong ecommerce growth. Stock prices surged after the merger announcement was made public.',
            'Entertainment': 'The blockbuster movie broke box office records on its opening weekend. The director received critical acclaim for the stunning screenplay and original soundtrack.',
        }
        sample_choice = st.selectbox('Or try a sample text:', ['Custom...'] + list(samples.keys()))
        input_text = st.text_area(
            'Document text:',
            value=samples.get(sample_choice, ''),
            height=180,
            placeholder='Paste any news article, paragraph, or document here...'
        )
    else:
        sample_col = text_col if text_col else 'text'
        if X_te_mat is not None and len(active_df) > 0:
            sample_rows = active_df[[text_col, label_col]].dropna().head(10)
            sample_opts = {f'Sample {i+1} ({row[label_col]})': row[text_col]
                           for i, (_, row) in enumerate(sample_rows.iterrows())}
            sample_choice = st.selectbox('Try a sample from your dataset:', ['Custom...'] + list(sample_opts.keys()))
            input_text = st.text_area('Document text:',
                                      value=sample_opts.get(sample_choice, ''),
                                      height=180,
                                      placeholder='Type or paste document text here...')
        else:
            input_text = st.text_area('Document text:', height=180,
                                      placeholder='Type or paste document text here...')

    if input_text and input_text.strip():
        clean_text = preprocess(input_text)
        X_vec      = active_vec.transform([clean_text])
        pred_label = active_clf.predict(X_vec)[0]
        pred_cat   = active_le.inverse_transform([pred_label])[0]
        col        = cat_colors.get(str(pred_cat), '#1E88E5')

        if hasattr(active_clf, 'decision_function'):
            scores = active_clf.decision_function(X_vec)[0]
            if len(CATEGORIES) == 2 and scores.ndim == 1:
                prob_pos = 1 / (1 + np.exp(-scores))
                probs    = np.array([1 - prob_pos, prob_pos])
            else:
                probs = np.exp(scores) / np.sum(np.exp(scores))
        elif hasattr(active_clf, 'predict_proba'):
            probs = active_clf.predict_proba(X_vec)[0]
        else:
            probs = np.zeros(len(active_le.classes_))
            probs[pred_label] = 1.0

        st.markdown('---')
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            card = (
                f'<div style="background:linear-gradient(135deg,{col}22,{col}11);'
                f'border:2px solid {col};border-radius:16px;padding:2rem;text-align:center;'
                f'box-shadow:0 6px 20px rgba(0,0,0,0.1)">'
                f'<div style="font-size:0.95rem;color:#555">Predicted Category</div>'
                f'<div style="font-size:3rem;font-weight:900;color:{col};margin:0.2rem 0">{pred_cat}</div>'
                f'<div style="background:{col};color:white;border-radius:25px;display:inline-block;'
                f'padding:0.3rem 1.4rem;font-weight:700;margin-top:0.3rem">Classified</div></div>'
            )
            st.markdown(card, unsafe_allow_html=True)

        st.markdown('---')
        st.markdown('#### Confidence Scores')
        conf_df = pd.DataFrame({'Category': active_le.classes_, 'Confidence': probs})
        conf_df = conf_df.sort_values('Confidence', ascending=False).reset_index(drop=True)
        for _, row in conf_df.iterrows():
            cat  = str(row['Category'])
            conf = row['Confidence']
            c    = cat_colors.get(cat, '#888')
            pct  = int(min(conf * 100, 100))
            is_top = (cat == str(pred_cat))
            st.markdown(
                f'<div style="margin:5px 0;display:flex;align-items:center;gap:10px">'
                f'<span style="min-width:140px;font-weight:{"700" if is_top else "400"};color:{col if is_top else "#444"}">{cat}</span>'
                f'<div style="flex:1;background:#eee;border-radius:8px;height:18px">'
                f'<div style="background:{c};width:{pct}%;height:100%;border-radius:8px;transition:width 0.3s"></div></div>'
                f'<span style="min-width:50px;text-align:right;font-size:0.88rem">{conf*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('---')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('#### Text Statistics')
            orig_w  = len(input_text.split())
            clean_w = len(clean_text.split())
            st.metric('Original Word Count',   orig_w)
            st.metric('After Preprocessing',   clean_w)
            st.metric('Stop Words Removed',    orig_w - clean_w)
            st.metric('Reduction %',           f'{(1-clean_w/max(orig_w,1))*100:.1f}%')
        with c2:
            st.markdown('#### Cleaned Text')
            st.text_area('Used for classification:', clean_text, height=130)

        st.markdown('---')
        st.markdown('#### Word Cloud of Input Document')
        if clean_text.strip():
            wc = WordCloud(width=700, height=280, background_color='white',
                           colormap='tab10', max_words=50, random_state=42).generate(clean_text)
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with MAIN_TABS[1]:
    if dataset_mode == 'Upload My Own Dataset' and user_df is not None:
        st.markdown('### Your Uploaded Dataset')

        disp_df = user_df[[text_col, label_col]].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total Documents', f'{len(disp_df):,}')
        c2.metric('Categories',      disp_df[label_col].nunique())
        c3.metric('Avg Text Length', f'{disp_df[text_col].str.split().str.len().mean():.0f} words')
        c4.metric('Missing Values',  disp_df.isnull().sum().sum())

        st.markdown('---')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('#### Category Distribution')
            counts = disp_df[label_col].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors  = [COLORS_LIST[i % len(COLORS_LIST)] for i in range(len(counts))]
            bars    = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor='white')
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        str(val), ha='center', fontweight='bold', fontsize=9)
            ax.set_title('Category Distribution', fontsize=13, fontweight='bold')
            ax.set_ylabel('Count')
            plt.xticks(rotation=30, ha='right', fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown('#### Category Pie Chart')
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(counts.values, labels=counts.index.astype(str),
                   autopct='%1.1f%%', colors=colors[:len(counts)],
                   startangle=90, textprops={'fontsize': 9})
            ax.set_title('Category Split', fontsize=13, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('---')
        st.markdown('#### Word Cloud per Category')
        df_proc = active_df.copy()
        df_proc['clean'] = preprocess_series(disp_df[text_col].fillna(''))
        n_cats = len(CATEGORIES)
        cols_wc = st.columns(min(4, n_cats))
        for i, cat in enumerate(CATEGORIES[:8]):
            cat_texts = df_proc[disp_df[label_col].values == cat]['clean']
            if len(cat_texts) > 0:
                combined = ' '.join(cat_texts.tolist())
                if combined.strip():
                    wc = WordCloud(width=300, height=200, background_color='white',
                                   colormap='tab10', max_words=40, random_state=42).generate(combined)
                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    ax.set_title(str(cat), fontsize=10, fontweight='bold',
                                 color=cat_colors.get(str(cat), '#333'))
                    plt.tight_layout()
                    cols_wc[i % 4].pyplot(fig); plt.close()

        st.markdown('---')
        st.markdown(f'#### Sample Data ({label_col} + {text_col})')
        st.dataframe(disp_df.head(15), use_container_width=True)

    else:
        st.markdown('### Default Dataset Analysis')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total Documents', '2,100')
        c2.metric('Categories', '7')
        c3.metric('Docs per Category', '300')
        c4.metric('Vectorizer', 'TF-IDF')
        st.markdown('---')
        if os.path.exists(PLOTS):
            plot_files = sorted([f for f in os.listdir(PLOTS) if f.endswith('.png')])
            eda_plots  = [p for p in plot_files if p.startswith(('01','02','03','08','10'))]
            cols = st.columns(2)
            for i, pf in enumerate(eda_plots):
                cols[i%2].markdown(f'#### {pf.split("_",1)[1].replace(".png","").replace("_"," ").title()}')
                cols[i%2].image(os.path.join(PLOTS,pf), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: NLP PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with MAIN_TABS[2]:
    st.markdown('### NLP Text Processing Pipeline')

    steps = [
        ('Step 1: Lowercase Conversion', 'Convert all text to lowercase.',
         '"AI Learning" -> "ai learning"'),
        ('Step 2: Remove Punctuation & Numbers', 'Remove non-alphabetic characters using regex.',
         '"3D-printing!" -> "D printing"'),
        ('Step 3: Tokenization (NLTK)', 'Split text into individual words.',
         '"machine learning" -> ["machine","learning"]'),
        ('Step 4: Stop Word Removal', 'Remove common English words (NLTK stopwords).',
         '"the data is important" -> ["data","important"]'),
        ('Step 5: Lemmatization (WordNet)', 'Reduce words to their base form.',
         '"running","runs","ran" -> "run"'),
        ('Step 6: Feature Extraction (TF-IDF)', 'Convert text to numerical TF-IDF vectors.',
         'Text -> [0.12, 0.05, 0.34, ...] (5000-dim vector)'),
        ('Step 7: Classification', 'ML classifier predicts document category.',
         'Vector -> "Technology"'),
    ]
    for title, desc, example in steps:
        with st.expander(title):
            st.write(f'**Description:** {desc}')
            st.code(f'Example: {example}')

    st.markdown('---')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### TF-IDF')
        st.markdown('TF-IDF(t,d) = TF(t,d) x log(N / df(t))')
        st.markdown('''
        - **TF**: Word frequency in document
        - **IDF**: Penalizes common words across docs
        - **Captures**: Rare but meaningful words
        ''')
    with col2:
        st.markdown('#### Bag of Words')
        st.markdown('''
        - Counts word occurrences
        - No importance weighting
        - Simple but effective baseline
        ''')
        st.code('"cat sat mat" -> {cat:1, sat:1, mat:1}')

    st.markdown('---')
    st.markdown('#### Upload Your Own Dataset Feature')
    st.markdown('''
    This app supports **custom dataset upload**. You can:
    - Upload any **CSV or Excel** file
    - Choose which column is the **document text**
    - Choose which column is the **category label**
    - The app trains **6 ML models** on your data **live**
    - Then classify any new document using your trained models

    **Supported formats:** `.csv`, `.xlsx`, `.xls`

    **Minimum requirements:** At least 10 documents, at least 2 categories.
    ''')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with MAIN_TABS[3]:
    st.markdown('### Model Performance Results')

    if dataset_mode == 'Upload My Own Dataset' and user_df is not None:
        st.info(f'Results on your uploaded dataset: **{len(user_df):,}** documents, **{len(CATEGORIES)}** categories')

    st.dataframe(results_df.reset_index(drop=True), use_container_width=True)

    if not results_df.empty:
        st.markdown('---')
        st.markdown('#### Accuracy Comparison')
        valid = results_df[results_df['Accuracy(%)'] > 0]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(valid['Model'], valid['Accuracy(%)'],
                       color=['#1E88E5','#43A047','#E53935','#8E24AA','#FB8C00','#00ACC1'],
                       edgecolor='white')
        ax.set_xlim(0, 115)
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, valid['Accuracy(%)']):
            ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                    f'{val}%', va='center', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    if dataset_mode == 'Upload My Own Dataset' and X_te_mat is not None and y_te_arr is not None:
        st.markdown('---')
        st.markdown('#### Confusion Matrix (Best Model)')
        try:
            cm  = confusion_matrix(y_te_arr, active_clf.predict(X_te_mat))
            fig, ax = plt.subplots(figsize=(max(6, len(CATEGORIES)), max(5, len(CATEGORIES)-1)))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=active_le.classes_,
                        yticklabels=active_le.classes_, ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {best_clf_name}', fontsize=13, fontweight='bold')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout(); st.pyplot(fig); plt.close()
        except Exception:
            pass
    elif dataset_mode == 'Use Default Dataset' and os.path.exists(PLOTS):
        for pf in ['06_confusion_matrix.png','04_tfidf_model_comparison.png','09_f1_comparison.png']:
            p = os.path.join(PLOTS, pf)
            if os.path.exists(p):
                st.markdown('---')
                st.markdown(f'#### {pf.split("_",1)[1].replace(".png","").replace("_"," ").title()}')
                st.image(p, use_container_width=True)

    st.markdown('---')
    st.markdown('### Why These NLP Techniques?')
    st.markdown('''
    | Technique | Reason |
    |-----------|--------|
    | **TF-IDF** | Captures word importance, not just frequency |
    | **Stop Word Removal** | Removes noise words |
    | **Lemmatization** | Reduces vocabulary, groups word variants |
    | **Naive Bayes** | Best for text classification (probabilistic) |
    | **SVM** | Optimal hyperplane for high-dimensional text data |
    | **Logistic Regression** | Fast, interpretable NLP baseline |
    ''')

st.markdown('---')
footer = (
    '<div style="text-align:center;color:#9e9e9e;padding:1.5rem;font-size:0.85rem;'
    'border-top:1px solid #eee">Document Classification using NLP | ML Minor Project | '
    '<strong>Shrawani Joshi</strong> | 2025-26</div>'
)
st.markdown(footer, unsafe_allow_html=True)
