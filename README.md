# Course Reviews - An NLP Approach
Building a pipeline for filtering, analyzing, and summarizing course reviews using modern ML techinques. 
## Web App
#### [Click here to use our Course Review Analyzer Streamlit App!](https://reviews-analyzer-bain.streamlit.app/)
## Overview
In this project, we do a deep dive on course reviews using NLP techniques. We analyze various classical NLP features as well as explore the use of text embeddings and LLMs to gauge the meaningfulness and sentiment of reviews.

#### KPIs
- Model can assess positive/negative sentiment better than random guessing (50/50)
- Pipeline can parse unstructured review inputs and predict the sentiment
- Filter out meaningless or gibberish entries, create app to allow user input
#### Stakeholders
- Course instructors who want to improve their courses or those in “voice of customer” type roles in industry
- Administrators in academia/industry who want to produce broad performance metrics for staff
- Students/customers who may be able to assess the quality of a course or product based on review summaries
## EDA
[Click here to see a more detailed exploration of the features.](eda/README.md)

We found a large database of [reviews of Coursera courses on Kaggle](https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera/data) that included details about the instructor, the course topic, the texts of real course reviews, and rating of the course from 1 to 5. 

We observed a significant class imbalance in the data (see below), where around 79% of the entries listed a 5/5 rating, however with millions of datapoints, we still had plenty of negative/neutral reviews to work with.

<img src="images/rating_hist.png" alt="drawing" width="400" margin='auto'/>. 

The reviews varied significantly in length as can be seen below. However, they also had widely varying *quality*. Here are a few below:

| | **Sample Reviews** | 
|-- | -- |
| 1 |A fantastic course for beginners. Explaining underlying concepts in a easy and understandable way. Dr. Chuck is fantastic. |
| 2 |GOOOOOOOOOOOOOOOOOOOOOOOd |
| 3 | Great course for beginners. I studied all programming fundamentals in school and was just trying to learn Python. I found that this course is very good for anyone that is trying to learn fundaments of programming even you have no prior knowledge. |
| 4 | A Great Course! |
| 5 | Fue una experiencia gratificante el poder realizar el curso. La flexibilidad que permite y la calidad de la información, merece la mejor calificación  | 

The varying quality led to ask an interesting question *Can we make a model that discriminates between meaningful and gibberish or meaningless reviews?"* Our goal is to gain actionable insights from the reviews. There were 29,031 reviews with < 5 characters and 78,044 with fewer than 10 characters. A few of these are shown below:

| | **Gibberish/Meaningless Revews** |
| -- | -- |
| 1 | jhkd |
| 2 | das | 
| 3 | Good..!! |
| 4 | T | 

The meaningless reviews were not limited to short random letters. Some were longer (>10 characters) sequences of random letters and some reviews contain real words, but contain no significant content. Although a review of "Good course" would indicate positive sentiment, it does not contain meaningful insights. This motivated our first and second sub-projects

1. Entropy analysis - High entropy can indicate nonsense text, but how do entropies differ by language? 
2. Gibberish detector - Can we create a model for identifiying meaningless text?

## Entropy Analysis
[Click here for more details on our entropy analysis](modeling/README.md)

In NLP, **entropy** measures the uncertainty or information content in a **language model's probability distribution** over possible next tokens or words.

$H(p) = -\sum\left[p(wᵢ) * \log₂ \left( p(wᵢ)\right)\right]$
- `H(p)`: Entropy of the probability distribution `p` over a vocabulary  
- `p(wᵢ)`: Probability assigned to word/token `wᵢ` by a language model  
- `n`: Total number of possible words/tokens in the vocabulary

This can be calcualted at the character, word, or sentence level. Gibberish text (e.g., "asdf asdf lkjweoiur qwe!") could tend to have high entropy because characters are random or nonsensical, there’s little repetition or pattern and because the distribution of characters is fairly uniform. R

## Gibberish Detector
[Click here for more details on our Gibberish detector](modeling/README.md)

We trained and tested a model for identifying meaningless reviews. Although it's hard to say for sure, we estimate that, similar to the dataset we were given, most reviews will not be gibberish/meaningless, so we compared against a baseline of always choosing the review was meaningful/not gibberish.

## Sentiment Analysis
[Click here for more details on our sentiment prediction model](modeling/README.md)


