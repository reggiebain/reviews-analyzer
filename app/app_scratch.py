import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate

# 🔹 Load Pretrained Models
@st.cache_resource()
def load_models():
    bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return bert_model, tokenizer, summarizer

bert_model, tokenizer, summarizer = load_models()

# 🔹 Load Classical ML Model (TF-IDF + Logistic Regression)
@st.cache_resource()
def load_tfidf_model():
    tfidf = TfidfVectorizer(max_features=5000)
    model = LogisticRegression()
    return tfidf, model

tfidf, classical_model = load_tfidf_model()

# 🔹 Load Sentence Embedding Model for Instructor Score
@st.cache_resource()
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

sentence_model = load_sentence_model()

# 🔹 Predict Sentiment with DistilBERT
def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"

# 🔹 Predict Sentiment with Classical ML
def predict_classical(text):
    vectorized = tfidf.transform([text])
    pred = classical_model.predict(vectorized)[0]
    return "Positive" if pred == 1 else "Negative"

# 🔹 Summarize Reviews
def summarize_reviews(reviews):
    if len(reviews) < 5:
        return "Not enough reviews to summarize."
    text = " ".join(reviews)
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# 🔹 Compute Course Score (Average Sentiment)
def compute_course_score(sentiments):
    return round((sum(sentiments) / len(sentiments)) * 100, 2) if sentiments else 0

# 🔹 Compute Instructor Score (Similarity to Positive Phrases)
def compute_instructor_score(reviews):
    positive_examples = ["The instructor is amazing", "Clear explanations", "Very knowledgeable"]
    review_embeddings = sentence_model.encode(reviews, convert_to_tensor=True)
    pos_embeddings = sentence_model.encode(positive_examples, convert_to_tensor=True)
    scores = [util.pytorch_cos_sim(review_embeddings, pos_embeddings[i]).mean().item() for i in range(len(positive_examples))]
    return round(np.mean(scores) * 100, 2)

# 🔹 Streamlit UI
st.title("📊 Course Review Analyzer")

# Text Input
review_text = st.text_area("Enter a course review:", "")

# Analyze Button
if st.button("Analyze Sentiment"):
    if review_text:
        # Predict Sentiment
        bert_sentiment = predict_bert(review_text)
        classical_sentiment = predict_classical(review_text)

        # Display Sentiment Results
        st.markdown("### **Sentiment Analysis Results:**")
        st.markdown(f"**🧠 DistilBERT Prediction:** {bert_sentiment}")
        st.markdown(f"**📊 Classical ML Prediction:** {classical_sentiment}")

        # Markdown Table
        table = [["DistilBERT", bert_sentiment], ["Classical ML", classical_sentiment]]
        st.markdown(tabulate(table, headers=["Model", "Prediction"], tablefmt="github"))
    else:
        st.warning("Please enter a review.")

# Bulk Analysis Section
st.markdown("---")
st.markdown("### 🔎 Bulk Analysis")

uploaded_file = st.file_uploader("Upload CSV file with reviews (column: 'review')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "review" in df.columns:
        reviews = df["review"].dropna().tolist()
        
        # Sentiment Analysis on all reviews
        sentiments = [1 if predict_bert(review) == "Positive" else 0 for review in reviews]
        avg_sentiment_score = compute_course_score(sentiments)
        avg_instructor_score = compute_instructor_score(reviews)
        summary_text = summarize_reviews(reviews)

        # Display Results
        st.markdown(f"### **📌 Course Score:** {avg_sentiment_score}%")
        st.markdown(f"### **🎓 Instructor Score:** {avg_instructor_score}%")
        st.markdown("### 📜 Review Summary:")
        st.write(summary_text)

        # Sentiment Distribution Plot
        fig, ax = plt.subplots()
        ax.hist(sentiments, bins=2, edgecolor="black", alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)
    
    else:
        st.error("CSV file must contain a 'review' column.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Transformers & Scikit-learn")
