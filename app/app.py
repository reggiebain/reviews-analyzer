import streamlit as st
import pandas as pd
import pickle
#import torch
from transformers import pipeline
from langdetect import detect
#import openai

# 🔹 Load your sentiment model (binary classifier)
import pathlib
import sys
dir = pathlib.Path.cwd().absolute()

sys.path.append(dir.parent.parent)
#sentiment_model_path = "app/sentiment_model_classical.pkl"
#with open(sentiment_model_path, "rb") as f:
#    sentiment_model = pickle.load(f)
'''
@st.cache_resource
def load_sentiment_model():
    with open("app/sentiment_model_classical.pkl", "rb") as f:
        return pickle.load(f)

sentiment_model = load_sentiment_model()


@st.cache_resource
def load_summarizer():
    #return pipeline("summarization", model="facebook/bart-large-cnn")
    #return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return pipeline("summarization", model="t5-small")

# 🔹 LLM feedback (you need an OpenAI API key)
openai.api_key = st.secrets.get("OPENAI_API_KEY")
'''
def load_sentiment_model():
    return pipeline("finiteautomata/bertweet-base-sentiment-analysis")
# 🔹 Example data
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
        return 0
    try:
        return detect(text)
    except:
        return 0

def predict_sentiment(text):
    return load_sentiment_model.predict([text])[0]  # binary: 0 = neg, 1 = pos
'''
def summarize_reviews_bullets(reviews):
    #summarizer = load_summarizer()
    combined_text = " ".join(reviews)
    #summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
    summary = "test summary"
    bullets_prompt = f"Turn the following review summary into 3 bullet points:\n\n{summary}"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": bullets_prompt}],
        temperature=0.7,
    )
    return completion.choices[0].message.content

def generate_constructive_feedback(reviews):
    joined_reviews = "\n".join(reviews)
    feedback_prompt = f"Here are some user reviews about a course:\n{joined_reviews}\n\nPlease provide constructive feedback for the course creator."
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": feedback_prompt}],
        temperature=0.7,
    )
    return completion.choices[0].message.content
'''
# Streamlit UI
st.title("📘 Course Review Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "review" in df.columns:
        reviews = df["review"].tolist()
    else:
        st.error("CSV must contain a 'review' column.")
        reviews = []
else:
    st.subheader("Or use sample reviews below:")
    reviews = st.text_area("Enter reviews (one per line)", "\n".join(sample_reviews)).split("\n")
    reviews = [r.strip() for r in reviews if r.strip()]

if reviews:
    results = []
    sentiments = []
    valid_reviews = []

    for review in reviews:
        if not can_detect_language(review):
            #sentiment = predict_sentiment(review)
            sentiment = 'test sentiment'
            results.append([review, sentiment])
            sentiments.append(sentiment)
            valid_reviews.append(review)
        else:
            results.append([review, "Ignored (non-English or invalid)"])

    df_results = pd.DataFrame(results, columns=["Review", "Sentiment (0=neg, 1=pos)"])
    st.write("### Sentiment Analysis Results")
    st.dataframe(df_results)

    if valid_reviews:
        # 🔹 Show 3-point summary
        st.write("### 🔍 Summary of Reviews (3 Bullet Points)")
        with st.spinner("Generating summary..."):
            bullet_summary = "test bullet summary"
            #bullet_summary = summarize_reviews_bullets(valid_reviews)
        st.markdown(bullet_summary)

        # 🔹 Compute score from 1 to 5
        avg_sentiment = sum(sentiments) / len(sentiments)
        score = round(1 + avg_sentiment * 4, 1)  # Scaled to 1–5
        st.write(f"### ⭐ Overall Course Rating: {score} / 5")

        # 🔹 Generate LLM feedback
        st.write("### 💡 Constructive Feedback for the Course Creator")
        with st.spinner("Asking LLM for feedback..."):
            feedback = 'test feedback'
            #feedback = generate_constructive_feedback(valid_reviews)
        st.markdown(feedback)
