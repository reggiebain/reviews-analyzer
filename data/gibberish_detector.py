import pickle
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import math
import re
import itertools
from langdetect import detect
import warnings
from tqdm import tqdm
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import os

# Suppress langdetect warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GibberishDetector:
    def __init__(self, model_path, ngram_ref=None, centroid=None):
        """Initialize with model path, n-gram reference, and optional centroid."""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.ngram_ref = ngram_ref or set()
        self.centroid = centroid  # Precomputed centroid for embeddings
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.xlm_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xlm_model.to(self.device)
        self.xlm_model.eval()
    
    # Entropy
    def calculate_entropy(self, text):
        if not isinstance(text, str) or pd.isna(text):
            return 0
        text = str(text).lower()
        length = len(text)
        if length == 0:
            return 0
        char_counts = Counter(text)
        entropy = -sum((count/length) * math.log2(count/length) for count in char_counts.values())
        return entropy
    
    # Language Detection
    def detect_language(self, text):
        if not isinstance(text, str) or pd.isna(text) or len(text.strip()) < 3:
            return 'unknown'
        try:
            return detect(text)
        except:
            return 'unknown'
    
    def cannot_detect_language(self, text):
        return 1 if text == 'unknown' else 0
    
    # Alphabet Detection
    def detect_alphabets(self, text):
        if not isinstance(text, str) or not text:
            return {
                'Chinese': {'present': False, 'count': 0},
                'Cyrillic': {'present': False, 'count': 0},
                'Hangul': {'present': False, 'count': 0},
                'Latin': {'present': False, 'count': 0}
            }
        
        ranges = {
            'Chinese': (0x4E00, 0x9FFF),
            'Cyrillic': (0x0400, 0x04FF),
            'Hangul': (0xAC00, 0xD7AF),
            'Latin': [(0x0000, 0x007F), (0x00A0, 0x00FF), (0x0100, 0x017F)]
        }
        
        alphabet_counts = defaultdict(int)
        for char in text:
            char_code = ord(char)
            if ranges['Chinese'][0] <= char_code <= ranges['Chinese'][1]:
                alphabet_counts['Chinese'] += 1
            if ranges['Cyrillic'][0] <= char_code <= ranges['Cyrillic'][1]:
                alphabet_counts['Cyrillic'] += 1
            if ranges['Hangul'][0] <= char_code <= ranges['Hangul'][1]:
                alphabet_counts['Hangul'] += 1
            for start, end in ranges['Latin']:
                if start <= char_code <= end:
                    alphabet_counts['Latin'] += 1
                    break
        
        return {
            'Chinese': {'present': alphabet_counts['Chinese'] > 0, 'count': alphabet_counts['Chinese']},
            'Cyrillic': {'present': alphabet_counts['Cyrillic'] > 0, 'count': alphabet_counts['Cyrillic']},
            'Hangul': {'present': alphabet_counts['Hangul'] > 0, 'count': alphabet_counts['Hangul']},
            'Latin': {'present': alphabet_counts['Latin'] > 0, 'count': alphabet_counts['Latin']}
        }
    
    # Word and Character Counts
    def word_count(self, text):
        if not isinstance(text, str):
            return 0
        words = re.split(r'\s+', text.strip())
        return len(words)
    
    def char_count(self, text):
        if not isinstance(text, str):
            return 0
        return len(text)
    
    def get_avg_word_length(self, text):
        if not isinstance(text, str):
            return 0
        words = re.split(r'\s+', text.strip())
        word_count = len(words)
        return sum(len(word) for word in words if word) / max(1, word_count) if words else 0
    
    # Repetition
    def get_max_repeated(self, text):
        if not isinstance(text, str):
            return 0
        return max([sum(1 for _ in g) for _, g in itertools.groupby(text)] or [0])
    
    # Punctuation Ratio
    def get_punct_ratio(self, text):
        if not isinstance(text, str):
            return 0
        char_length = len(text)
        punct_count = sum(1 for c in text if c in '.,!?')
        return punct_count / max(1, char_length)
    
    # N-gram Coherence
    def get_ngram_coherence(self, text, n=2):
        if not isinstance(text, str):
            return 0
        text_lower = text.lower()
        total_ngrams = max(1, len(text_lower) - n + 1)
        valid_ngrams = sum(1 for i in range(total_ngrams) if text_lower[i:i+n] in self.ngram_ref)
        return valid_ngrams / total_ngrams
    
    # Embedding Features
    def get_embeddings(self, texts, batch_size=32):
        """Generate XLM-Roberta embeddings for a list of texts."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.xlm_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def compute_centroid(self, df, review_col='review', label_col='is_gibberish', sample_size=10000):
        """Compute centroid from real reviews (static for training)."""
        real_texts = df[df[label_col] == 0][review_col].dropna().tolist()
        if len(real_texts) > sample_size:
            real_texts = np.random.choice(real_texts, sample_size, replace=False).tolist()
        embeddings = self.get_embeddings(real_texts)
        return np.mean(embeddings, axis=0)
    
    def get_embedding_features(self, text, embed_cache=None):
        """Get cosine similarity and anomaly score for a single text."""
        if not isinstance(text, str):
            text = ''
        embedding = self.get_embeddings([text])[0]
        if self.centroid is None:
            return 0.0, 0.0  # No centroid provided
        cosine_sim = cosine_similarity([embedding], self.centroid.reshape(1, -1))[0][0]
        anomaly_score = np.linalg.norm(embedding - self.centroid)
        return cosine_sim, anomaly_score
    
    # Feature Extraction
    def extract_features(self, text):
        """Extract all features for a single text."""
        if not isinstance(text, str):
            text = ''
        
        alphabets = self.detect_alphabets(text)
        lang = self.detect_language(text)
        cosine_sim, anomaly_score = self.get_embedding_features(text)
        
        return [
            self.calculate_entropy(text),
            self.cannot_detect_language(lang),
            int(alphabets['Chinese']['present']),
            int(alphabets['Cyrillic']['present']),
            int(alphabets['Hangul']['present']),
            int(alphabets['Latin']['present']),
            self.word_count(text),
            self.char_count(text),
            self.get_avg_word_length(text),
            self.get_max_repeated(text),
            self.get_punct_ratio(text),
            self.get_ngram_coherence(text),
            cosine_sim,
            anomaly_score
        ]
    
    def predict(self, text):
        """Predict if text is gibberish (0) or real (1)."""
        features = self.extract_features(text)
        return self.model.predict([features])[0]
    
    def predict_df(self, df, review_col='review', embed_path=None):
        """Predict gibberish for a DataFrame column with progress bar."""
        tqdm.pandas(desc=f"Extracting gibberish features from '{review_col}'...")
        df_clean = df.copy()
        df_clean[review_col] = df_clean[review_col].fillna('').astype(str)
        
        # Base features (non-embedding)
        features = df_clean[review_col].progress_apply(lambda x: [
            self.calculate_entropy(x),
            self.cannot_detect_language(self.detect_language(x)),
            int(self.detect_alphabets(x)['Chinese']['present']),
            int(self.detect_alphabets(x)['Cyrillic']['present']),
            int(self.detect_alphabets(x)['Hangul']['present']),
            int(self.detect_alphabets(x)['Latin']['present']),
            self.word_count(x),
            self.char_count(x),
            self.get_avg_word_length(x),
            self.get_max_repeated(x),
            self.get_punct_ratio(x),
            self.get_ngram_coherence(x)
        ])
        features_df = pd.DataFrame(features.tolist(), index=df.index, columns=[
            'entropy', 'cannot_detect_language', 'has_chinese', 'has_cyrillic',
            'has_hangul', 'has_latin', 'word_count', 'n_chars', 'avg_word_length',
            'max_repeated', 'punct_ratio', 'ngram_coherence'
        ])
        
        # Embedding features
        if embed_path and os.path.exists(embed_path):
            print(f"Loading embeddings from {embed_path}")
            embeddings = np.load(embed_path)
        else:
            texts = df_clean[review_col].tolist()
            embeddings = self.get_embeddings(texts)
            if embed_path:
                np.save(embed_path, embeddings)
                print(f"Saved embeddings to {embed_path}")
        
        cosine_sim = cosine_similarity(embeddings, self.centroid.reshape(1, -1)).flatten()
        anomaly_score = np.linalg.norm(embeddings - self.centroid, axis=1)
        
        features_df['cosine_to_centroid'] = cosine_sim
        features_df['anomaly_score'] = anomaly_score
        
        return self.model.predict(features_df)

    @staticmethod
    def build_ngram_reference(texts, n=2, top_k=1000, sample_size=10000):
        """Build n-gram reference from a list of texts."""
        if len(texts) > sample_size:
            texts = np.random.choice(texts, sample_size, replace=False)
        ngrams = Counter()
        for text in tqdm(texts, desc="Building n-gram reference..."):
            text = str(text).lower()
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                if not ngram.isspace():
                    ngrams[ngram] += 1
        return set([ngram for ngram, _ in ngrams.most_common(top_k)])