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
<img src="images/rating_hist.png" alt="drawing" width="250"/>. 

The reviews varied significantly in length as can be seen below. However, they also had widely varying *quality*. Here are a few below:

| | **Samples of Review** | 
|-- | -- |
| 1 |A fantastic course for beginners. Explaining underlying concepts in a easy and understandable way. Dr. Chuck is fantastic. |
| 2 |GOOOOOOOOOOOOOOOOOOOOOOOd |
| 3 | Great course for beginners. I studied all programming fundamentals in school and was just trying to learn Python. I found that this course is very good for anyone that is trying to learn fundaments of programming even you have no prior knowledge. |
| 4 | A Great Course! |
| 5 | Fue una experiencia gratificante el poder realizar el curso. La flexibilidad que permite y la calidad de la información, merece la mejor calificación  | 




