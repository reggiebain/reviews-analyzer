# Note - Script ignores char counts, centroid calcualtions, and anomaly scores for simplicity.
# These were not super important features anyway and they are complicated to calculate

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scipy.stats as stats
import re
import nltk
from nltk.corpus import words
from collections import Counter
import math
import unicodedata
import time
from tqdm import tqdm
import pickle
from langdetect import detect
import warnings
import pycountry
from scipy import stats
from collections import defaultdict
import itertools
import os
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import wordfreq

class gibberish_detector:
    def __init__(self, ngram_ref_path, centroid_path):
        # Load gibberish detection model
        with open(gibberish_model_path, "rb") as f:
            self.gibberish_model = pickle.load(f)

        # Load sentiment model
        with open(sentiment_model_path, "rb") as f:
            self.sentiment_model = pickle.load(f)
            
# Load in sentiment data with sentiment features calcualted
def load_data(FILE_PATH):
    df = pd.read_csv(FILE_PATH) #load data
    df = df.dropna(subset=['review'])# clean out null entries
    return df

# FEATURE - Calculate entropy
def calculate_entropy(text):
    """Calculate Shannon entropy of the text to detect randomness."""
    if not text:
        return 0
    if not isinstance(text, str) or pd.isna(text):
        return 0  # Return 0 for NaN or non-string values
    text = str(text).lower()
    length = len(text)
    if length == 0:  # Handle empty strings
        return 0
    char_counts = Counter(text)
    entropy = -sum((count/length) * math.log2(count/length) for count in char_counts.values())
    return entropy

def create_entropy_feature(df, review_col='review'):
    tqdm.pandas(desc='Calculating entropies: ')
    df['entropy'] = df['review'].progress_apply(calculate_entropy)
    return df

def detect_language(text):
    if not isinstance(text, str) or pd.isna(text) or len(text.strip()) < 3:
        return 'unknown'  # For NaN, empty, or very short text
    try:
        return detect(text)
    except:
        return 'unknown'  # Fallback for any detection errors

# Returns 0 if we can't find the langauge, 1 if we can
def cannot_detect_language(text):
    if text == 'unknown':
        return 1
    else:
        return 0

def create_can_detect_feature(df, review_col='review'):
    tqdm.pandas(desc="Detecting Language...")
    df['language'] = df[review_col].progress_apply(detect_language)
    df['cannot_detect_language'] = df['language'].progress_apply(cannot_detect_language)
    return df

def word_count(text):
    words = re.split(f'\s+', text.strip())
    word_count = len(words)
    return word_count

def create_word_and_char_counts_feature(df, review_col='review'):
    tqdm.pandas(desc='Getting word counts...')
    df['word_count'] = df[review_col].progress_apply(word_count)
    return df

def get_avg_word_length(text):
    # avg word length
    words = re.split(f'\s+', text.strip())
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words if word) / max(1, word_count) if words else 0
    return avg_word_length

def create_avg_word_length_feature(df, review_col='review'):
    tqdm.pandas(desc='Getting avg word length feature...')
    df['avg_word_length'] = df[review_col].progress_apply(get_avg_word_length)
    return df

# FEATURE - Amount of Reptition
def get_max_repeated(text):
    max_repeats = max([sum(1 for _ in g) for _, g in itertools.groupby(text)] or [0])
    return max_repeats

def create_repetition_feature(df, review_col='review'):
    tqdm.pandas(desc='Creating repetition feature...')
    df['max_repeated'] = df[review_col].progress_apply(get_max_repeated)
    return df

def get_punct_ratio(text):
    char_length = len(text)
    punct_count = sum(1 for c in text if c in '.,!?')
    punct_ratio = punct_count / max(1, char_length)
    return punct_ratio
    
def create_punct_ratio_feature(df, review_col='review'):
    tqdm.pandas(desc='Creating punctuation ratio feature...')
    df['punct_ratio'] = df[review_col].progress_apply(get_punct_ratio)
    return df

# FEATURE - Contains common n-grams
# Step 1 - Build n-gram reference from sample of real reviews
def build_ngram_reference(texts, n=2, top_k=1000, sample_size=10000):

    # Sample texts to avoid over-processing (e.g., 1.19M reviews)
    if len(texts) > sample_size:
        texts = np.random.choice(texts, sample_size, replace=False)
    
    # Generate n-grams
    ngrams = Counter()
    for text in tqdm(texts, desc="Building n-gram reference..."):
        text = str(text).lower()
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            if not ngram.isspace():
                ngrams[ngram] += 1
    
    # Return top k most common n-grams
    return set([ngram for ngram, _ in ngrams.most_common(top_k)])

# FEATURE - ngram coherence, fraction of ngrams that appear in list of common ngrams
def get_ngram_coherence(text, n=2):
    text_lower = text.lower()
    total_ngrams = max(1, len(text_lower) - n + 1)
    valid_ngrams = sum(1 for i in range(total_ngrams) if text_lower[i:i+n] in ngram_ref)
    ngram_coherence = valid_ngrams / total_ngrams
    return ngram_coherence

def create_ngram_coherence_feature(df, ngram_ref, review_col='review'):
    tqdm.pandas(desc='Calcualting ngram coherenece...')
    df['ngram_coherence'] = df[review_col].progress_apply(get_ngram_coherence)
    return df

# Function to get embeddings in batches
def get_embeddings(texts, batch_size=32):
    embeddings = []

    # Load pre-trained XLM-R model and tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    embeddings_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_model.to(device)
    embeddings_model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = embeddings_model(**inputs)
        # Use [CLS] token embedding (first token)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)
    
# Compute centroid from substantive reviews
def compute_centroid(df, review_col='review', label_col='is_gibberish', sample_size=10000):
    # Use real reviews (training data only)
    real_texts = df[df[label_col] == 0][review_col].dropna().tolist()
    if len(real_texts) > sample_size:
        real_texts = np.random.choice(real_texts, sample_size, replace=False).tolist()
    embeddings = get_embeddings(real_texts)
    return np.mean(embeddings, axis=0)

# Add embedding-based features
def add_embedding_features(df, centroid, review_col='review', embed_path=None):
    if embed_path and os.path.exists(embed_path):
        print(f"Loading embeddings from {embed_path}")
        embeddings = np.load(embed_path)
    else:
        texts = df[review_col].fillna('').tolist()
        embeddings = get_embeddings(texts)
        if embed_path:
            np.save(embed_path, embeddings)
            print(f"Saved embeddings to {embed_path}")
    
    # Cosine similarity to centroid
    cosine_sim = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    # Anomaly score (Euclidean distance)
    anomaly_score = np.linalg.norm(embeddings - centroid, axis=1)
    
    df['cosine_to_centroid'] = cosine_sim
    df['anomaly_score'] = anomaly_score
    return df

def create_feature_df(df, review_col='review', ngram_ref=None, centroid=None):
    df = create_entropy_feature(df, review_col='review')
    df = create_can_detect_feature(df, review_col='review')
    df = create_word_and_char_counts_feature(df, review_col='review')
    df = create_avg_word_length_feature(df, review_col='review')
    df = create_repetition_feature(df, review_col='review')
    df = create_punct_ratio_feature(df, review_col='review')
    df = create_ngram_coherence_feature(df, ngram_ref, review_col='review')
    df = add_embedding_features(df, centroid)
    return df

def main():
# %%
# Load precomputed centroid and ngram_ref from Amazon full training set
with open('/kaggle/input/reviews-analyzer-dataset/coursera_gibberish/ngram_ref.pkl', 'rb') as f:
    ngram_ref = pickle.load(f)
with open('/kaggle/input/reviews-analyzer-dataset/coursera_gibberish/centroid.pkl', 'rb') as f:
    centroid = pickle.load(f)
    
# Step 3: Create feature DataFrames
features = create_feature_df(df, ngram_ref=ngram_ref, centroid=centroid)

# %%
train_features.columns

# %%
#sns.histplot(data=train_features, x='neg_word_count')
train_features[['pos_word_count', 'neg_word_count', 'negated_pos_count',
       'negated_neg_count', 'pos_ngram_count', 'neg_ngram_count',
       'polarity_score', 'exclamation_count', 'uppercase_ratio',
       'sentiment', 'entropy',
       'word_count', 'n_chars', 'avg_word_length', 'max_repeated',
       'punct_ratio', 'ngram_coherence', 'cosine_to_centroid',
       'anomaly_score']].hist(figsize=(15, 10))
plt.suptitle("Distributions of Sentiment and Gibberish Features - Coursera Reviews", fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('dists_of_sentiment_and_gibberish_features.png')
plt.show()

# %%
# Choose the relevant columns
model_features = ['cannot_detect_language', 'entropy', 'word_count', 'avg_word_length',
                  'ngram_coherence', 'anomaly_score', 'punct_ratio', 'max_repeated']
feed_into_model_df = train_features.reindex(columns=model_features)

# Run the model on coursera stuff
model_path = '/kaggle/input/reviews-analyzer-dataset/gibberish_random_forest_model.pkl'
with open(model_path, 'rb') as f:
    gibberish_model = pickle.load(f)

train_filtered_df = train_features.copy()
train_filtered_df['is_gibberish'] = gibberish_model.predict(feed_into_model_df)
train_filtered_df['gibberish_probs'] = gibberish_model.predict_proba(feed_into_model_df)[:,1]

# %%
train_filtered_df.is_gibberish.value_counts()

# %% [markdown]
# Looks like way too many reviews are getting tagged as gibberish. At first glance, looks like shorter reviews could be the issue. Let's investigate and then do some model pruning.

# %% [markdown]
# ## Investigating What is Classified as "Gibberish"

# %% [markdown]
# ### Word Count and Polarity

# %%
#train_filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
inf_columns = train_filtered_df.columns[(train_filtered_df == np.inf).any()]
print("Columns with inf values:", inf_columns.tolist())

# %%
# Investigate word counts vs. is gibberish
fig, axes = plt.subplots(1,2, figsize=(8,4))
sns.histplot(data=train_filtered_df, x='word_count', hue='is_gibberish', ax=axes[0])
sns.boxplot(data=train_filtered_df, x="is_gibberish", y='word_count', ax=axes[1])
plt.suptitle(f'Word Count Distribution (Sample of n={sample_size} Reviews)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('word_count_gibberish.png')

# %% [markdown]
# So it's pretty clear this is a major issue. Nearly all of the gibberish entries are super short word count wise. Let's look at all of the entries with fewer than 15 words. But if we check some of these out.

# %%
short_gibb_entries = train_filtered_df[(train_filtered_df['word_count']<=15) & (train_filtered_df['is_gibberish'] == 1)]
short_entries = train_filtered_df[(train_filtered_df['word_count']<=15)]
gibb_entries = train_filtered_df[(train_filtered_df['is_gibberish'] == 1)]
print(f"| Stat (N={sample_size} Sample) | Value | ")
print(f"| -- | -- |")
print(f"| No. Short Entries that are Gibberish | {len(short_gibb_entries)} |")
print(f"| No. Short Entries Total | {len(short_entries)}|")
print(f"| Pct Short Entries that are Gibberish | {len(short_gibb_entries)/len(short_entries):.2%}|")
print(f"| Pct Gibberish Entries that are Short | {len(short_gibb_entries)/len(gibb_entries):.2%}|")


# %% [markdown]
# | Stat (N=1000 Sample) | Value | 
# | -- | -- |
# | No. Short Entries (Gibberish) | 311 |
# | No. Short Entries Total | 540|
# | Pct Short Entries that are Gibberish | 57.59%|
# | Pct Gibberish Entries that are Short | 100.00%|

# %%
short_reviews_df = train_filtered_df[train_filtered_df['word_count'] <= 15]
fig, axes = plt.subplots(2,2, figsize=(10,8))
sns.scatterplot(data = short_reviews_df, x='word_count', y='gibberish_probs', hue='is_gibberish', ax=axes[0][0])
sns.scatterplot(data=short_reviews_df, x='polarity_score', y='gibberish_probs', hue='is_gibberish', ax=axes[0][1])
sns.scatterplot(data=short_reviews_df, x='avg_word_length', y='gibberish_probs', hue='is_gibberish', ax=axes[1][0])
sns.scatterplot(data=short_reviews_df, x='entropy', y='gibberish_probs', hue='is_gibberish', ax=axes[1][1])
plt.suptitle(f'Gibberish Prob. vs. Important Features w/ Word Length <= 15 (Sample n={sample_size})', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('gibberish_probs_vs_features.png')

# %% [markdown]
# Word count is clearly an issue. How can we identify short word counts that are NOT really gibberish. Lots look like "great course" or "thanks" or "good one". How can we otherwise flag these? We could look at polarity, entropy, etc. Let's look at those. Recall that polarity is (Per chatgpt):
# 
# **Interpreting Polarity Scores**
# - Polarity ≈ 0 → Neutral sentiment (balanced positive/negative or no strong sentiment)
# - Polarity > 0 → Positive sentiment (more positive words/phrases than negative)
# - Polarity < 0 → Negative sentiment (more negative words/phrases than positive)
# 
# **Specific values:**
# - 0: Completely neutral (e.g., "This is a course.").
# - 1: Slightly positive (e.g., "Decent course.").
# - 2–3: Clearly positive (e.g., "Good and useful course!").
# - 5+: Strongly positive (e.g., "Absolutely amazing and fantastic!").
# - -1: Slightly negative (e.g., "Not great.").
# - -2 to -3: Clearly negative (e.g., "Boring and useless.").
# - <-5: Extremely negative (e.g., "Horrible, worst ever!").

# %%
# Look at entries with neutral polarity and low word count
short_reviews_df[(short_reviews_df['is_gibberish'] == 1) & (short_reviews_df['polarity_score'] == 0)].sample(n=10).review.to_markdown()
print(f"Pct. of Short Gibberish Reviews w/ Polarity 0: {(len(short_reviews_df[(short_reviews_df['is_gibberish'] == 1) & (short_reviews_df['polarity_score'] == 0)].review))/sample_size:.2%}")

# %% [markdown]
# 
# | Short Reviews (<15 words) Marked Gibberish and Polarity = 0  |
# |:-------------------------------------------------|
# | Very Nourishing.                                 |
# | simple and informative                           |
# | Approche claire et très pédagogique              |
# | Need some more practical examples on sub modules |
# | .                                                |
# | _                                                |
# | COMPLETO Y MUY APLICATIVO                        |
# | Very informative course.timely needed one.       |
# | Some code can not work.                          |
# | basico, pero bien.                               |

# %%
short_reviews_df[(short_reviews_df['is_gibberish'] == 1) & (short_reviews_df['polarity_score']== 1)].sample(n=10).review.to_markdown()
print(f"Pct. of Short Gibberish Reviews w/ Polarity 1: {(len(short_reviews_df[(short_reviews_df['is_gibberish'] == 1) & (short_reviews_df['polarity_score'] == 1)].review))/sample_size:.2%}")

# %% [markdown]
# | Short Reviews (<15 words) Marked Gibberish with Polarity = 1 | 
# |:-------------------------------------------------------------------|
# | Excellent course!                                                  |
# | Excellent course!                                                  |
# | Excelente Curso, practico y didactico.                             |
# | Excellent course I did on coursera!!!!                             |
# | Very useful websites I learned AWS machine learning                |
# | great course                                                       |
# | One of the best course ever😇😇😇😇                                |
# | great introduction to topics necessary for a IT help desk position |
# | Very good course                                                   |
# | thank you so much for the great course                             |

# %% [markdown]
# **High Polarity Reviews**
# - Although many of these short, high polarity reviews are not very "substantive" (meaning we can't really draw any insights from them) they are reflective of strong positive feelings.
# 
# **Neutral Polarity Reviews**
# - The polarity 0 results that are short and marked as gibberish are a bit less interesting. They don't convey strong positive/negative feelings AND they are not substantive. So what insights can we draw from them?
# 
# **Note**
# - Although the original intent of a gibberish detector was to identify meaningless text, I think we have an opportunity to classify meaningful vs. non-meaningful reviews as well

# %% [markdown]
# ### Classifier Probability Cutoff

# %%
cutoff = 0.90
train_filtered_df['is_gibberish_high_cutoff'] = train_filtered_df['gibberish_probs'] > cutoff
train_filtered_df[train_filtered_df['is_gibberish_high_cutoff'] == 1].review

# %%
cutoffs = np.linspace(0.5, 1, 50)  # 50 cutoff points from 0 to 1

# Initialize lists to store results
num_gibberish = []
avg_polarity = []
avg_word_count = []
avg_entropy = []

for cutoff in cutoffs:
    gibberish_reviews = train_filtered_df[train_filtered_df['gibberish_probs'] >= cutoff]
    
    num_gibberish.append(len(gibberish_reviews))
    avg_polarity.append(gibberish_reviews['polarity_score'].mean() if not gibberish_reviews.empty else np.nan)
    avg_word_count.append(gibberish_reviews['review'].str.split().str.len().mean() if not gibberish_reviews.empty else np.nan)
    avg_entropy.append(gibberish_reviews['entropy'].mean() if not gibberish_reviews.empty else np.nan)

# %%
fig, axes = plt.subplots(2, 2, figsize=(12,8))
axes[0][0].plot(cutoffs, num_gibberish, marker='o', linestyle='-')
axes[0][0].set_xlabel("Probability Cutoff")
axes[0][0].set_ylabel("Number of Gibberish Reviews")
axes[0][0].set_title(f"Gibberish Classification vs. Cutoff (Sample of {sample_size})")
axes[0][0].grid(True)

# Plot Average Polarity vs. Cutoff
axes[0][1].plot(cutoffs, avg_polarity, marker='o', linestyle='-', color='r')
axes[0][1].set_xlabel("Probability Cutoff")
axes[0][1].set_ylabel("Average Polarity of Gibberish")
axes[0][1].set_title(f"Polarity of Gibberish Reviews vs. Cutoff (Sample of {sample_size})")
axes[0][1].grid(True)

# Plot Average Word Count vs. Cutoff
axes[1][0].plot(cutoffs, avg_word_count, marker='o', linestyle='-', color='g')
axes[1][0].set_xlabel("Probability Cutoff")
axes[1][0].set_ylabel("Average Word Count of Gibberish")
axes[1][0].set_title(f"Word Count of Gibberish Reviews vs. Cutoff (Sample of {sample_size})")
axes[1][0].grid(True)

# Plot Average Word Entropy vs. Cutoff
axes[1][1].plot(cutoffs, avg_entropy, marker='o', linestyle='-', color='orange')
axes[1][1].set_xlabel("Probability Cutoff")
axes[1][1].set_ylabel("Average Entropy of Gibberish")
axes[1][1].set_title(f"Entropy of Gibberish Reviews vs. Cutoff (Sample of {sample_size})")
axes[1][1].grid(True)

plt.tight_layout()
plt.savefig('gibberish_vs_cutoffs_multi_var.png')

# %% [markdown]
# ### Hybrid Approach with Model + Word Validity Check
# - We can try thresholds, where it looks like we'll get mostly low entropy, low word count, low polarity actual gibberish (or meaningless reviews) if we simply up our threshold to say 0.9 instead of 0.5.
# - However, we could also check to see if reviews have at least a few real words.

# %%
import nltk
from nltk.tokenize import word_tokenize
from wordfreq import word_frequency

# Set minimum word frequency for a word to be considered "real"
FREQUENCY_THRESHOLD = 1e-6  # Adjust based on experimentation

# Function to check if a word is commonly used in a given language
def is_real_word(word, lang, wordlist='large'):
    try:
        return word_frequency(word, lang, minimum=0, wordlist=wordlist) > FREQUENCY_THRESHOLD
    except LookupError:
        return 0

# Updated function using detected language
def adjust_is_gibberish(row, length_threshold=5, word_ratio_threshold=0.75):
    review = row["review"]
    detected_lang = row['language']
    
    # If language is undetectable, assume gibberish (optional rule)
    if row["cannot_detect_language"]:
        return True  # Consider these gibberish

    # Tokenize the review
    tokens = word_tokenize(review.lower())
    tokens = [w for w in tokens if w.isalpha()]  # Remove non-alphabetic tokens
    
    # Identify meaningful words using word frequency data
    meaningful_words = [word for word in tokens if is_real_word(word, detected_lang)]
    
    # Allow very short but meaningful reviews (e.g., "Excellent!", "Great course!")
    if len(tokens) < length_threshold and len(meaningful_words) == len(tokens):
        return False
    
    # If the proportion of meaningful words is too low, consider it gibberish
    if len(meaningful_words) / max(len(tokens), 1) < word_ratio_threshold:
        return True

    return False

# Apply function to each row using the detected language column
tqdm.pandas(desc='Adjusting Cutoff and Finding real words...')
CUTOFF = 0.90
train_filtered_df['is_gibberish'] = train_filtered_df['gibberish_probs'] > CUTOFF
train_filtered_df["is_gibberish"] = train_filtered_df.progress_apply(adjust_is_gibberish, axis=1)

# %%
print(f"After Real World Filtering - Pct Marked As Gibberish = {train_filtered_df['is_gibberish'].sum()/len(train_filtered_df['is_gibberish']):.2%}")

# %%
wordfreq.available_languages()

# %% [markdown]
# ## Apply to Full Train, Val and Test Sets
# - The below can probably be used as a script later

# %%
# Load in sentiment data with sentiment features calcualted
train_df = pd.read_pickle('/kaggle/input/reviews-analyzer-dataset/sentiment_data/sentiment_train.pkl')
val_df = pd.read_pickle('/kaggle/input/reviews-analyzer-dataset/sentiment_data/sentiment_val.pkl')
test_df = pd.read_pickle('/kaggle/input/reviews-analyzer-dataset/sentiment_data/sentiment_test.pkl')

# Recombine becuase we forgot to not split before
file_path = '/kaggle/working/embeddings.npy'
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File '{file_path}' has been deleted.")
else:
    print(f"File '{file_path}' does not exist.")

# Clean out entries with no review
train_df = train_df.dropna(subset=['review'])
val_df = val_df.dropna(subset=['review'])
test_df = test_df.dropna(subset=['review'])

# %%
print(f"Training Size = {len(train_df)}")
print(f"Validation Size = {len(val_df)}")
print(f"Testing Size = {len(test_df)}")

# %%
SAMPLE_FRAC = 0.25
# Re grab all of the featurs for full training, val, and test sets
train_features = create_feature_df(train_df.sample(frac=SAMPLE_FRAC), ngram_ref=ngram_ref, centroid=centroid)
val_features = create_feature_df(val_df.sample(frac=SAMPLE_FRAC), ngram_ref=ngram_ref, centroid=centroid)
test_features = create_feature_df(test_df.sample(frac=SAMPLE_FRAC), ngram_ref=ngram_ref, centroid=centroid)

# Choose the relevant columns
model_features = ['cannot_detect_language', 'entropy', 'word_count', 'avg_word_length',
                  'ngram_coherence', 'anomaly_score', 'punct_ratio', 'max_repeated']
feed_train_into_model_df = train_features.reindex(columns = model_features)
feed_val_into_model_df = test_features.reindex(columns=model_features)
feed_test_into_model_df = val_features.reindex(columns=model_features)

# Run the model trained on amazon reviews on coursera stuff
model_path = '/kaggle/input/reviews-analyzer-dataset/gibberish_random_forest_model.pkl'
with open(model_path, 'rb') as f:
    gibberish_model = pickle.load(f)

# Filter the FULL train set
train_filtered_df = train_features.copy()
train_filtered_df['is_gibberish'] = gibberish_model.predict(feed_train_into_model_df)
train_filtered_df['gibberish_probs'] = gibberish_model.predict_proba(feed_train_into_model_df)[:,1]
train_filtered_df['is_gibberish'] = train_filtered_df['gibberish_probs'] > CUTOFF
train_filtered_df["is_gibberish"] = train_filtered_df.apply(adjust_is_gibberish, axis=1)

# Filter the val set
val_filtered_df = val_features.copy()
val_filtered_df['is_gibberish'] = gibberish_model.predict(feed_val_into_model_df)
val_filtered_df['gibberish_probs'] = gibberish_model.predict_proba(feed_val_into_model_df)[:,1]
val_filtered_df['is_gibberish'] = val_filtered_df['gibberish_probs'] > CUTOFF
val_filtered_df["is_gibberish"] = val_filtered_df.apply(adjust_is_gibberish, axis=1)

# Filter the test set
test_filtered_df = test_features.copy()
test_filtered_df['is_gibberish'] = gibberish_model.predict(feed_test_into_model_df)
test_filtered_df['gibberish_probs'] = gibberish_model.predict_proba(feed_test_into_model_df)[:,1]
test_filtered_df["is_gibberish"] = test_filtered_df.apply(adjust_is_gibberish, axis=1)

# %% [markdown]
# ### Export Data

# %%
'''
import pickle
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed

# Load the trained model
model_path = '/kaggle/input/reviews-analyzer-dataset/gibberish_random_forest_model.pkl'
with open(model_path, 'rb') as f:
    gibberish_model = pickle.load(f)

# Define batch size and save path
BATCH_SIZE = 5000  # Adjust as needed
SAVE_PATH = "processed_batches.pkl"  # Stores progress

# Function to process a single batch
def process_batch(batch_df, batch_idx, ngram_ref, centroid, cutoff=0.5):
    """Process a batch and save output."""
    batch_features = create_feature_df(batch_df, ngram_ref=ngram_ref, centroid=centroid)

    model_features = ['cannot_detect_language', 'entropy', 'word_count', 'avg_word_length',
                      'ngram_coherence', 'anomaly_score', 'punct_ratio', 'max_repeated']
    
    batch_features = batch_features.reindex(columns=model_features)

    # Predict gibberish
    batch_features['is_gibberish'] = gibberish_model.predict(batch_features)
    batch_features['gibberish_probs'] = gibberish_model.predict_proba(batch_features)[:, 1]
    batch_features['is_gibberish'] = batch_features['gibberish_probs'] > cutoff
    batch_features["is_gibberish"] = batch_features.apply(adjust_is_gibberish, axis=1)

    # Save batch result
    batch_file = f"batch_{batch_idx}.pkl"
    batch_features.to_pickle(batch_file)

    print(f"✅ Saved batch {batch_idx} -> {batch_file}")

# Function to track progress and resume if interrupted
def batch_process_data(df, batch_size, ngram_ref, centroid, cutoff=0.5, num_jobs=4):
    """Process data in parallel batches with resume capability."""
    num_batches = int(np.ceil(len(df) / batch_size))
    processed_batches = set()

    # Load previously completed batches
    if os.path.exists(SAVE_PATH):
        processed_batches = pickle.load(open(SAVE_PATH, "rb"))
        print(f"🔄 Resuming from saved progress... {len(processed_batches)} batches completed")

    # Process only unprocessed batches
    unprocessed_batches = [i for i in range(num_batches) if i not in processed_batches]

    def process_and_save(batch_idx):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end]

        process_batch(batch_df, batch_idx, ngram_ref, centroid, cutoff)

        # Update progress file
        processed_batches.add(batch_idx)
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(processed_batches, f)

    # Use parallel processing
    Parallel(n_jobs=num_jobs)(delayed(process_and_save)(i) for i in unprocessed_batches)

    # Combine all batch files into final dataframe
    batch_files = [f"batch_{i}.pkl" for i in range(num_batches) if os.path.exists(f"batch_{i}.pkl")]
    train_filtered_df = pd.concat([pd.read_pickle(f) for f in batch_files], ignore_index=True)

    return train_filtered_df

# Run parallelized batch processing
train_filtered_df = batch_process_data(train_df, BATCH_SIZE, ngram_ref, centroid, CUTOFF, num_jobs=4)
'''

# %%
import zipfile

# Save as pickle files
output_dir = '/kaggle/working/'  # Kaggle default; adjust for local (e.g., './data/')
os.makedirs(output_dir, exist_ok=True)

files_to_save = {
    'sentiment_train_clean.pkl': train_filtered_df,
    'sentiment_val_clean.pkl': val_filtered_df,
    'sentiment_test_clean.pkl': test_filtered_df,
}

print("Saving datasets as pickle files...")
for filename, obj in tqdm(files_to_save.items(), desc="Saving Pickle Files"):
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(obj, f)

print(f"Saved pickle files to {output_dir}: {list(files_to_save.keys())}")

# Zip the pickle files
zip_filename = os.path.join(output_dir, 'clean_sentiment_data.zip')
print("Zipping files...")
with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for filename in tqdm(files_to_save.keys(), desc="Adding Files to Zip"):
        zipf.write(os.path.join(output_dir, filename), arcname=filename)

print(f"Saved and zipped files to {zip_filename}")

# %%



