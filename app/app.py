import streamlit as st
import pandas as pd
from transformers import pipeline
import torch
from huggingface_hub import login
from openai import OpenAI
import re

# get open ai client using my secret
openai_client = OpenAI(api_key=st.secrets['api_keys']['OPENAI_API_KEY'])

# Custom CSS for dark mode-friendly styling
st.markdown("""
    <style>
    .title {
        color: #4CAF50; /* Bright green for title */
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .delimiter {
        border: 2px solid #81C784; /* Lighter green for lines */
        margin: 10px 0;
    }
    .subheader {
        color: #81C784; /* Lighter green for subheaders */
        font-size: 24px;
        font-weight: bold;
    }
    .warning-box {
        background-color: #FF8F00; /* Vibrant orange background */
        color: #FFFFFF; /* White text for contrast */
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #FFB300; /* Bright orange border */
    }
    .result-box {
        background-color: #2E7D32; /* Dark green background */
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #81C784; /* Light green border */
        margin-top: 10px; 
        width: 100%; 
    }
    </style>
""", unsafe_allow_html=True)

# Title with delimiter
st.markdown('<div class="title">Course Review Analyzer</div>', unsafe_allow_html=True)
st.markdown('<hr class="delimiter">', unsafe_allow_html=True)

# Make two main columns, left is info and questions, right is results
left_col, right_col = st.columns([1,1.5])

with left_col:
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(
        "images/app_logo.png",
        caption="How do students feel about your class?",
        width=250
    )
    st.markdown('</div>', unsafe_allow_html=True)
with right_col:
    st.markdown('<div class="subheader">App Description</div>', unsafe_allow_html=True)
    #st.markdown('<hr class="delimiter">', unsafe_allow_html=True)
    st.write("""This app uses NLP techniques to analyze course reviews. 
             It can track the sentiment, provide a summary, give you an overall score, 
             and provide constructive feedback based on your reviews.""")
    st.markdown("""
            **Models Used:**
            - *Sentiment Analysis*: `nlptown/bert-base-multilingual-uncased-sentiment` (Hugging Face)
            - *Summarization*: `t5-small` (Hugging Face)
            - *Rating*: Calculated locally from sentiment results
            - *Feedback*: `gpt-3.5-turbo` (OpenAI)
        """, unsafe_allow_html=True)

# Sample reviews (hardcoded for simplicity; could be loaded from a file)
st.markdown('<div class="subheader">Reviews</div>', unsafe_allow_html=True)
#st.markdown('<hr class="delimiter">', unsafe_allow_html=True)
sample_reviews = [
    "This course was amazing, I learned so much!",
    "Terrible experience, the instructor was unprepared.",
    "It was okay, nothing special but not bad either.",
    "Loved the practical examples, really helpful!",
    "Waste of time, content was outdated."
]

# Display sample reviews
reviews = st.text_area("Enter reviews (one per line) or use the sample reviews shown", 
                       "\n".join(sample_reviews), height=200).split("\n")
reviews = [r.strip() for r in reviews if r.strip()]

# Buttons
st.markdown('<div class="subheader">Actions</div>', unsafe_allow_html=True)
st.write("Click on an option below. *Note* that summary/feedback should only be run after analyzing sentiment")
#st.markdown('<hr class="delimiter">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    analyze_clicked = st.button("Analyze Sentiment", use_container_width=True)
with col2:
    summarize_clicked = st.button("Summarize Reviews", use_container_width=True)
with col3:
    rating_clicked = st.button("Get Overall Rating", use_container_width=True)
with col4:
    feedback_clicked = st.button("Get Feedback", use_container_width=True)

# Store sentiment results or summary in session state for reuse
if "sentiment_results" not in st.session_state:
    st.session_state.sentiment_results = None

if "summary" not in st.session_state:
    st.session_state.summary = None

@st.cache_resource
def load_sentiment_analyzer():
    # Login to Hugging Face with debug output
    try:
        st.write("Logging in to Hugging Face...")
        login(token=st.secrets['api_keys']['HF_TOKEN'])
        st.write("Hugging Face login successful.")
    except KeyError:
        st.error("Hugging Face token not found in secrets. Please configure it in Streamlit Cloud settings.")
        st.stop()
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        st.stop()
    try:
        return pipeline(
            'sentiment-analysis',
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1
        )
    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            st.error("Too many requests to Hugging Face Hub. Please wait a moment and try again.")
        else:
            st.error(f"Failed to load sentiment model: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to load sentiment model: {str(e)}")
        return None

@st.cache_resource
def load_summarizer():
    try:
        return pipeline(
            "summarization",
            model='t5-small',
            device=-1
        )
    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            st.error("Too many requests to Hugging Face Hub. Please wait a moment and try again.")
        else:
            st.error(f"Failed to load summarization model: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to load summarization model: {str(e)}")
        return None

#st.markdown('<div class="subheader">Results</div>', unsafe_allow_html=True)
#st.markdown('<hr class="delimiter">', unsafe_allow_html=True)
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
                    "Review": reviews,
                    "Sentiment": [result["label"].capitalize() for result in results],
                    "Confidence": [round(result["score"], 2) for result in results]
                })
                # Store results in "session state"
                st.session_state.sentiment_results = df

                # Display results
                st.markdown('<div class="subheader">Sentiment Analysis Results</div>', unsafe_allow_html=True)
                st.write("Here’s the sentiment for each review:")
                st.table(df)  # Plain table, no green box
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
                # Code for running summarizer with HF
                #summary = summarizer(joined_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
                
                # Code for running Open AI
                prompt = f"Summarize the following course reviews in a concise paragraph:\n\n{joined_text}"
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a concise summarizer."},
                              {"role": "user", "content": prompt}
                              ],
                    max_tokens=100,
                    temperature=0.5
                )
                summary = response.choices[0].message.content.strip()

                st.session_state.summary = summary
                #bullet_points = summary.split(". ")[:3]
                bullet_points = re.split(f'[.!]', summary)
                st.markdown('<div class="subheader">Summary of Reviews</div>', unsafe_allow_html=True)
                st.markdown("\n".join([f"- {point.strip()}" for point in bullet_points if point.strip()]))
            except Exception as e:
                st.error(f"Summarization failed: {str(e)}")

# ----------------- RATING TASK -------------------
if rating_clicked:
    if st.session_state.sentiment_results is None:
        st.markdown('<div class="warning-box">Please run "Analyze Sentiment" first to get sentiment results!</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="subheader">Overall Course Rating</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="subheader">Constructive Feedback</div>', unsafe_allow_html=True)
                st.write(feedback)
            except Exception as e:
                st.error(f"Failed to get feedback from LLM: {str(e)}")

# Footer
st.markdown('<hr class="delimiter">', unsafe_allow_html=True)
st.write("App designed by Reggie Bain. Powered by Hugging Face Transformers, OpenAI and Streamlit.")
