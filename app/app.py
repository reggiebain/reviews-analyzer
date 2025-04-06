import streamlit as st
import pandas as pd
from transformers import pipeline
import torch
from huggingface_hub import login
from openai import OpenAI

# Login to HG using token stored in secrets
login(token=st.secrets['api_keys']['HF_TOKEN'])

# get open ai client using my secret
openai_client = OpenAI(api_key=st.secrets['api_keys']['OPENAI_API_KEY'])

# Title and description
st.title("Course Review Sentiment Analyzer")
st.write("This app analyzes the sentiment of course reviews, summarizes them, and provides an overall rating. Use the buttons below!")

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
col1, col2, col3, col4 = st.columns(4)
with col1:
    analyze_clicked = st.button("Analyze Sentiment")

with col2:
    summarize_clicked = st.button("Summarize Reviews")

with col3:
    rating_clicked = st.button("Get Overall Rating")

with col4:
    feedback_clicked = st.button('Get Feedback')

# Store sentiment results or summary in session state for reuse
if "sentiment_results" not in st.session_state:
    st.session_state.sentiment_results = None

if "summary" not in st.session_state:
    st.session_state.summary = None

@st.cache_resource
def load_sentiment_analyzer():
    try:
        return pipeline(
            'sentiment-analysis',
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1
        )
    except Exception as e:
        st.error(f"Failed to load sentiment model: {str(e)}")
        return None

# Cache summarization model
@st.cache_resource
def load_summarizer():
    try:
        return pipeline(
            "summarization",
            #model="sshleifer/distilbart-cnn-12-6",
            model='t5-small',
            device=-1
        )
    except Exception as e:
        st.error(f"Failed to load summarization model: {str(e)}")
        return None

# ----------------- ANALYSIS TASK --------------------
if analyze_clicked:
    with st.spinner("Loading model and analyzing sentiments..."):
        sentiment_analyzer = load_sentiment_analyzer()
        if sentiment_analyzer is None:
            st.error("Sentiment analysis failed due to model loading issue.")
        else:
            try:
                #sentiment_analyzer = pipeline('sentiment-analysis', 
                #                              model="nlptown/bert-base-multilingual-uncased-sentiment",
                #                              device=-1)
                results = sentiment_analyzer(reviews)

                # Prepare data for table
                df = pd.DataFrame({
                    "Review": sample_reviews,
                    "Sentiment": [result["label"].capitalize() for result in results],
                    "Confidence": [round(result["score"], 2) for result in results]
                })
                # Store results in "session state"
                st.session_state.sentiment_results = df

                # Display results
                st.subheader("Sentiment Analysis Results")
                st.write("Here’s the sentiment for each review:")
                st.table(df)
            except Exception as e:
                st.error(f"Sentiment analysis failed: {str(e)}")

# ----------------- SUMMARIZE TASK -------------------
if summarize_clicked:
    with st.spinner("Loading summarization model and generating summary..."):
        summarizer = load_summarizer()
        if summarizer is None:
            st.error("Summarization failed due to model loading issue.")
        else:
            try:
                joined_text = " ".join(reviews)
                summary = summarizer(joined_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
                st.session_state.summary = summary
                bullet_points = summary.split(". ")[:3]
                st.subheader("Summary of Reviews")
                st.markdown("\n".join([f"- {point.strip()}" for point in bullet_points if point.strip()]))
            except Exception as e:
                st.error(f"Summarization failed: {str(e)}")

# ----------------- RATING TASK -------------------
if rating_clicked:
    if st.session_state.sentiment_results is None:
        st.warning("Please run 'Analyze Sentiment' first to get sentiment results!")
    else:
        with st.spinner("Calculating overall rating..."):
            df = st.session_state.sentiment_results

            # Map star ratings that come out of the NLP town model above to numerical scores (1 to 5)
            star_map = {
                "1 star": 1.0,
                "2 stars": 2.0,
                "3 stars": 3.0,
                "4 stars": 4.0,
                "5 stars": 5.0
            }
            scores = [star_map.get(sent, 3.0) * conf for sent, conf in zip(df["Sentiment"], df["Confidence"])]  # Default to 3.0 if unexpected label

            # Calculate average score and scale to 1-5 (already in this range)
            avg_score = sum(scores) / len(scores) if scores else 3.0  # Default to 3 if no scores
            rating = round(avg_score, 1)

            # Generate justification based on star counts
            star_counts = df["Sentiment"].value_counts().to_dict()
            justification_parts = [f"{star_counts.get(st, 0)} were {st}" for st in star_map.keys()]
            justification = f"Out of {len(df)} reviews, " + ", ".join([part for part in justification_parts if part[0] != "0"]) + "."

            # Display rating and justification
            st.subheader("Overall Course Rating")
            st.write(f"The course is rated: **{rating} / 5**")
            st.write(f"Justification: {justification}")

# ----------------- FEEDBACK TASK -------------------
if feedback_clicked:
    if st.session_state.summary is None:
        st.warning("Please run 'Summarize Reviews' first to generate a summary!")
    else:
        with st.spinner("Generating constructive feedback from LLM..."):
            try:
                summary = st.session_state.summary
                prompt = (
                    f"Here is a summary of course reviews: '{summary}'. "
                    "Please provide constructive feedback on how the course could be improved based on this summary."
                )
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful course instructor providing constructive feedback."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                feedback = response.choices[0].message.content.strip()
                st.subheader("Constructive Feedback")
                st.write(feedback)
            except Exception as e:
                st.error(f"Failed to get feedback from LLM: {str(e)}")

# Footer
st.write("Powered by Hugging Face Transformers and Streamlit.")
