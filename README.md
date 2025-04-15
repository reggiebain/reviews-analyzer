# Course Reviews - An NLP Approach
#### *Building a pipeline for filtering, analyzing, and summarizing course reviews using modern ML and NLP techinques* 
## Web App
#### [Click here to use our Course Review Analyzer Streamlit App!](https://reviews-analyzer-bain.streamlit.app/)
## Overview
In this project, we do a deep dive on course reviews using NLP techniques. We analyze various classical NLP features as well as explore the use of text embeddings and LLMs to gauge the meaningfulness and sentiment of reviews.

#### KPIs
- Model can assess positive/negative sentiment better than random guessing (50/50) or other relevant baselines.
- Pipeline can parse unstructured review inputs and predict the sentiment
- Filter out meaningless or gibberish entries, create app to allow user input

#### Stakeholders
- Course instructors who want to improve their courses or those in “voice of customer” type roles in industry
- Administrators in academia/industry who want to produce broad performance metrics for staff
- Students/customers who may be able to assess the quality of a course or product based on review summaries
## EDA
[Click here for a detailed discussion and visualizations of our full exploratory data analysis.](./eda_feature_extraction/README.md)

We studied a large set of Coursera course reviews from Kaggle as well as a set of Amazon product reviews where *gibberish* reviews were labeled. [Click here](./data_and_saved_models/README.md) for additional discussion of our datasets. 

## Modeling
[Click here for detailed discussion of our model building and results](./modeling_and_results/README.md)

The discussion in our EDA section [linked here](./eda_feature_extraction/README.md) helped to motivate our first and second sub-projects described below:

#### 1. Entropy analysis
- High entropy can indicate nonsense text, but how do entropies differ by language? 
- Notebooks: [Entropy Analysis Notebook](./modeling_and_results/entropy-stats-analysis.ipynb)

#### 2. Gibberish detector 
- Can we create a model for identifiying meaningless text?
- Notebooks: [Gibberish Feature Building (Amazon Reviews)](./eda_feature_extraction/gibberish-classifier-build-features.ipynb), [Coursera Gibberish Feature Extraction](./eda_feature_extraction/coursera-extract-gibberish-features-nonscript.ipynb)

Once we culled the reviews for quality reviews from which we could get actionable insights, our third and final project (for now) was analyzing sentiment:

#### 3. Sentiment Analyzer 
- Experiment with classical NLP features and a feature-based ML approach as well as fine-tuning a modern pre-trained deep learning model.
- Notebooks: [Extract Sentiment Features](./eda_feature_extraction/sentiment-analyzer-feature-extraction.ipynb), [Sentiment Model Building](./modeling_and_results/sentiment-analysis-model.ipynb)



