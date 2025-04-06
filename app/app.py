import streamlit as st
import pandas as pd
from transformers import pipeline
import torch
from huggingface_hub import login

#HF_TOKEN = st.secrets['api_keys']['HF_TOKEN']
login(token=st.secrets['api_keys']['HF_TOKEN'])

# Title and description
st.title("Course Review Sentiment Analyzer")
st.write("This app analyzes the sentiment of course reviews. Click 'Analyze' to see the results!")

# Sample reviews (hardcoded for simplicity; could be loaded from a file)
sample_reviews = [
    "This course was amazing, I learned so much!",
    "Terrible experience, the instructor was unprepared.",
    "It was okay, nothing special but not bad either.",
    "Loved the practical examples, really helpful!",
    "Waste of time, content was outdated."
]

# Display sample reviews
st.subheader("Sample Reviews")
st.write("Here are some example course reviews (positive and negative):")
reviews = st.text_area("Enter reviews (one per line)", "\n".join(sample_reviews)).split("\n")
reviews = [r.strip() for r in reviews if r.strip()]

# Display some buttons side by side
col1, col2, col3 = st.columns(3)
with col1:
    analyze_clicked = st.button("Analyze Sentiment")

with col2:
    summarize_clicked = st.button("Summarize Reviews")

with col3:
    rating_clicked = st.button("Get Overall Rating")

# Store sentiment results in session state for reuse
if "sentiment_results" not in st.session_state:
    st.session_state.sentiment_results = None

# ----------------- ANALYSIS TASK --------------------
if analyze_clicked:
    # Load sentiment analysis pipeline from Hugging Face
    # Using a lightweight model for faster deployment (distilbert)
    with st.spinner("Loading model and analyzing sentiments..."):

        sentiment_analyzer = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
        results = sentiment_analyzer(reviews)

        # Prepare data for table
        df = pd.DataFrame({
            "Review": sample_reviews,
            "Sentiment": [result["label"].capitalize() for result in results],
            "Confidence": [round(result["score"], 2) for result in results]
        })

        # Display results
        st.subheader("Sentiment Analysis Results")
        st.write("Here’s the sentiment for each review:")
        st.table(df)

# ----------------- SUMMARIZE TASK -------------------
if summarize_clicked:
    with st.spinner("Loading summarization model and generating summary..."):
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        joined_text = " ".join(reviews)
        summary = summarizer(joined_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]

        # Break summary into 3 bullet points (simple split for readability)
        bullet_points = summary.split(". ")[:3]
        st.subheader("Summary of Reviews")
        st.markdown("\n".join([f"- {point.strip()}" for point in bullet_points if point.strip()]))
        
# ----------------- RATING TASK -------------------
if rating_clicked:
    if st.session_state.sentiment_results is None:
        st.warning("Please run 'Analyze Sentiment' first to get sentiment results!")
    else:
        with st.spinner("Calculating overall rating..."):
            df = st.session_state.sentiment_results

            # Map sentiments to scores: Positive = 1, Neutral = 0.5, Negative = 0
            sentiment_scores = {
                "Positive": 1.0,
                "Neutral": 0.5,
                "Negative": 0.0
            }
            scores = [sentiment_scores[sent] * conf for sent, conf in zip(df["Sentiment"], df["Confidence"])]

            # Calculate average score and scale to 1-5
            avg_score = sum(scores) / len(scores) if scores else 0
            rating = round(1 + 4 * avg_score, 1)  # 1 (all Negative) to 5 (all Positive)

            # Generate justification
            pos_count = len(df[df["Sentiment"] == "Positive"])
            neg_count = len(df[df["Sentiment"] == "Negative"])
            neu_count = len(df[df["Sentiment"] == "Neutral"])
            total = len(df)
            justification = (
                f"Out of {total} reviews, {pos_count} were Positive, "
                f"{neg_count} were Negative, and {neu_count} were Neutral."
            )

            # Display rating and justification
            st.subheader("Overall Course Rating")
            st.write(f"The course is rated: **{rating} / 5**")
            st.write(f"Justification: {justification}")
# Footer
st.write("Powered by Hugging Face Transformers and Streamlit.")
