import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

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
for review in sample_reviews:
    st.write(f"- {review}")

# Button to trigger analysis
if st.button("Analyze"):
    # Load sentiment analysis pipeline from Hugging Face
    # Using a lightweight model for faster deployment (distilbert)
    with st.spinner("Loading model and analyzing sentiments..."):
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU-only for Streamlit Cloud
        )

        # Analyze sentiment for each review
        results = sentiment_analyzer(sample_reviews)

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

# Footer
st.write("Powered by Hugging Face Transformers and Streamlit.")