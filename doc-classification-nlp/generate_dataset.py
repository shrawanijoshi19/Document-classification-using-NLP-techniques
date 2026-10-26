# coding: utf-8
"""
Generate synthetic News Document dataset for classification.
Categories: Technology, Sports, Politics, Entertainment, Business, Health, Science
Author : Shrawani Joshi
Project: ML Minor Project - Document Classification using NLP
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

CATEGORIES = ['Technology', 'Sports', 'Politics', 'Entertainment', 'Business', 'Health', 'Science']

# ── Per-category word pools ────────────────────────────────────────────────────
WORD_POOLS = {
    'Technology': [
        'software', 'hardware', 'artificial', 'intelligence', 'machine', 'learning',
        'neural', 'network', 'algorithm', 'data', 'cloud', 'computing', 'cybersecurity',
        'blockchain', 'internet', 'programming', 'python', 'developer', 'app', 'mobile',
        'robot', 'automation', 'digital', 'innovation', 'startup', 'semiconductor',
        'processor', 'database', 'server', 'encryption', 'deep', 'model', 'training',
        'dataset', 'framework', 'open', 'source', 'github', 'deployment', 'API',
        'microservice', 'kubernetes', 'docker', 'tensorflow', 'pytorch', 'GPU', 'CPU',
        'quantum', 'computing', 'virtual', 'reality', 'augmented', 'metaverse', 'chip',
        'silicon', 'transistor', 'bandwidth', 'latency', 'protocol', 'fiber', 'optic'
    ],
    'Sports': [
        'football', 'cricket', 'tennis', 'basketball', 'soccer', 'match', 'tournament',
        'player', 'team', 'coach', 'championship', 'goal', 'score', 'stadium', 'athlete',
        'olympic', 'medal', 'gold', 'silver', 'bronze', 'league', 'cup', 'final', 'winner',
        'defeat', 'victory', 'penalty', 'referee', 'training', 'fitness', 'injury', 'record',
        'sprint', 'marathon', 'swimming', 'cycling', 'boxing', 'wrestling', 'badminton',
        'volleyball', 'hockey', 'rugby', 'transfer', 'contract', 'season', 'pitch', 'court',
        'race', 'driver', 'lap', 'podium', 'Formula', 'grand', 'prix', 'batting', 'bowling'
    ],
    'Politics': [
        'government', 'election', 'parliament', 'president', 'minister', 'policy', 'democracy',
        'vote', 'party', 'opposition', 'bill', 'law', 'senate', 'congress', 'constitution',
        'diplomacy', 'treaty', 'sanction', 'military', 'budget', 'tax', 'reform', 'protest',
        'campaign', 'candidate', 'ballot', 'corruption', 'judicial', 'court', 'verdict',
        'legislation', 'amendment', 'foreign', 'affairs', 'summit', 'alliance', 'debate',
        'referendum', 'coalition', 'cabinet', 'secretary', 'ambassador', 'embassy', 'crisis',
        'ceasefire', 'negotiation', 'humanitarian', 'refugee', 'border', 'sovereignty', 'regime'
    ],
    'Entertainment': [
        'movie', 'film', 'actor', 'actress', 'director', 'cinema', 'music', 'song', 'album',
        'concert', 'festival', 'award', 'oscars', 'grammy', 'celebrity', 'television', 'show',
        'series', 'episode', 'streaming', 'netflix', 'disney', 'box', 'office', 'blockbuster',
        'trailer', 'premiere', 'review', 'critic', 'audience', 'ticket', 'stage', 'performance',
        'dance', 'comedy', 'drama', 'thriller', 'romance', 'animation', 'sequel', 'franchise',
        'producer', 'script', 'screenplay', 'soundtrack', 'chart', 'single', 'band', 'singer',
        'rapper', 'genre', 'release', 'fans', 'viral', 'trending', 'social', 'media', 'influencer'
    ],
    'Business': [
        'company', 'market', 'stock', 'shares', 'profit', 'revenue', 'investment', 'merger',
        'acquisition', 'startup', 'entrepreneur', 'CEO', 'board', 'shareholders', 'quarterly',
        'earnings', 'growth', 'economy', 'finance', 'bank', 'loan', 'interest', 'rate', 'trade',
        'export', 'import', 'supply', 'demand', 'inflation', 'recession', 'GDP', 'fiscal',
        'monetary', 'currency', 'forex', 'commodity', 'oil', 'gold', 'retail', 'ecommerce',
        'consumer', 'brand', 'marketing', 'advertising', 'campaign', 'sales', 'partnership',
        'contract', 'deal', 'valuation', 'IPO', 'venture', 'capital', 'equity', 'debt', 'bond'
    ],
    'Health': [
        'medicine', 'hospital', 'doctor', 'patient', 'treatment', 'disease', 'virus', 'vaccine',
        'drug', 'clinical', 'trial', 'symptom', 'diagnosis', 'surgery', 'therapy', 'mental',
        'health', 'wellbeing', 'nutrition', 'diet', 'exercise', 'obesity', 'diabetes', 'cancer',
        'heart', 'blood', 'pressure', 'pandemic', 'epidemic', 'WHO', 'research', 'medical',
        'pharmaceutical', 'antibiotic', 'immune', 'system', 'gene', 'therapy', 'stem', 'cell',
        'organ', 'transplant', 'ambulance', 'emergency', 'ICU', 'nurse', 'pharmacy', 'dosage',
        'chronic', 'acute', 'pathology', 'radiology', 'MRI', 'scan', 'laboratory', 'test'
    ],
    'Science': [
        'research', 'experiment', 'discovery', 'universe', 'planet', 'galaxy', 'NASA', 'space',
        'rocket', 'satellite', 'telescope', 'physics', 'chemistry', 'biology', 'evolution',
        'fossil', 'climate', 'atmosphere', 'ocean', 'earthquake', 'volcano', 'biodiversity',
        'species', 'genetic', 'DNA', 'protein', 'molecule', 'atom', 'electron', 'quantum',
        'particle', 'accelerator', 'laboratory', 'hypothesis', 'theory', 'proof', 'publication',
        'journal', 'peer', 'review', 'Nobel', 'prize', 'astronaut', 'Mars', 'Moon', 'orbit',
        'gravitational', 'wave', 'black', 'hole', 'supernova', 'comet', 'asteroid', 'telescope'
    ]
}

TEMPLATES = [
    'The {noun1} of {noun2} has significantly impacted the world of {cat}.',
    'Experts in {cat} are discussing the latest development in {noun1} and {noun2}.',
    'A major breakthrough in {noun1} was announced this week in the field of {cat}.',
    'The {noun2} sector has seen tremendous growth due to advances in {noun1}.',
    'Reports indicate that {noun1} is reshaping how {noun2} works in modern {cat}.',
    'Scientists and professionals in {cat} are excited about new findings in {noun1}.',
    'New {noun1} strategies are being adopted by leaders in the {cat} industry.',
    'The role of {noun2} in {cat} cannot be understated, especially with recent {noun1} trends.',
    'A comprehensive study on {noun1} reveals surprising insights about {cat} practices.',
    'International {cat} communities are rallying around the issue of {noun1} and {noun2}.',
]

def make_document(category, min_words=60, max_words=120):
    pool = WORD_POOLS[category]
    template = np.random.choice(TEMPLATES)
    n1 = np.random.choice(pool)
    n2 = np.random.choice(pool)
    sentence = template.format(noun1=n1, noun2=n2, cat=category.lower())
    # pad with random words from pool + fillers
    fillers = ['the', 'is', 'are', 'was', 'were', 'has', 'have', 'been', 'with',
               'from', 'this', 'that', 'which', 'also', 'its', 'their', 'new',
               'major', 'key', 'recent', 'important', 'significant', 'several',
               'various', 'further', 'increasing', 'leading', 'latest', 'global']
    n_words  = np.random.randint(min_words, max_words)
    extra    = [np.random.choice(pool + fillers) for _ in range(n_words)]
    doc      = sentence + ' ' + ' '.join(extra) + '.'
    return doc

def generate_dataset(n_per_class=300):
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_class):
            rows.append({'text': make_document(cat), 'category': cat})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_dataset()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'documents.csv')
    df.to_csv(out, index=False)
    print(f'Dataset saved -> {out}')
    print(f'Shape: {df.shape}')
    print(df['category'].value_counts())
    print(df.head(3))
