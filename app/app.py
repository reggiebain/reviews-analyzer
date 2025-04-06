import streamlit as st
import pandas as pd
import pickle
import torch
from transformers import pipeline
from langdetect import detect

# Load sentiment analysis model
sentiment_model_path = "sentiment_model.pkl"
with open(sentiment_model_path, "rb") as f:
    sentiment_model = pickle.load(f)

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample reviews
sample_reviews = [
    "This course was very informative and well-structured. I learned a lot!",
    "The content was good, but the pacing was a bit too fast for me.",
    "Not worth the money. The instructor was not engaging at all.",
    "Loved it! The examples were very practical and easy to follow.",
    "Decent course, but I expected more depth on certain topics.",
    "Too much theory and not enough hands-on practice.",
    "Amazing course! It really helped me improve my skills.",
    "The quizzes were too difficult and not well explained.",
    "I appreciate the effort, but the course was a bit outdated.",
    "Solid introduction to the topic, but could use more real-world examples."
]

def can_detect_language(text):
    if not isinstance(text, str) or pd.isna(text) or len(text.strip()) < 3:
        return 0  # For NaN, empty, or very short text
    try:
        return detect(text)
    except:
        return 0  # Fallback for any detection errors

# Function to predict sentiment
def predict_sentiment(text):
    return sentiment_model.predict([text])[0]  # Returns sentiment score (0-2)

# Function to summarize reviews
def summarize_reviews(reviews):
    combined_text = " ".join(reviews)
    summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

# Streamlit UI
st.title("Course Review Sentiment Analyzer")

# File uploader or manual input
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "review" in df.columns:
        reviews = df["review"].tolist()
    else:
        st.error("CSV must contain a 'review' column.")
        reviews = []
else:
    st.subheader("Enter Reviews Manually or Use Sample Reviews")
    reviews = st.text_area("Enter reviews (one per line)", "\n".join(sample_reviews)).split("\n")
    reviews = [r.strip() for r in reviews if r.strip()]

if reviews:
    results = []
    sentiments = []
    valid_reviews = []

    for review in reviews:
        if not can_detect_language(review):
            sentiment = predict_sentiment(review)
            results.append([review, sentiment])
            sentiments.append(sentiment)
            valid_reviews.append(review)
        else:
            results.append([review, "Gibberish (ignored)"])

    df_results = pd.DataFrame(results, columns=["Review", "Predicted Sentiment (0-2)"])
    st.write("### Sentiment Analysis Results")
    st.dataframe(df_results)

    if valid_reviews:
        summary = summarize_reviews(valid_reviews)
        st.write("### Summary of Reviews")
        st.write(summary)

        avg_sentiment = sum(sentiments) / len(sentiments)
        st.write(f"### Overall Rating: {round(avg_sentiment, 2)}")
