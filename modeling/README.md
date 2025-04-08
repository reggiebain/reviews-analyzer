# Modeling
## Entropy Statistical Analysis

## Gibberish/Meaningful Review Classifier

## Sentiment Prediction Model

| Model               |   Accuracy |   F1 Score | Best Hyperparameters                        |
|---------------------|------------|------------|---------------------------------------------|
| Logistic Regression |   0.941255 |   0.919223 | {'C': 0.01}                                 |
| Random Forest       |   0.940048 |   0.913632 | {'max_depth': 5, 'n_estimators': 50}        |
| Gradient Boosting   |   0.938975 |   0.909423 | {'learning_rate': 0.01, 'n_estimators': 50} |


| Model                   |   Accuracy |   F1 Score |
|-------------------------|------------|------------|
| Distilbert-base-uncased |    0.78581 |    0.83025 |

25% Run
| Model               |   Accuracy |   F1 Score | Best Hyperparameters                        |
|---------------------|------------|------------|---------------------------------------------|
| Logistic Regression |   0.941296 |   0.919062 | {'C': 0.01}                                 |
| Random Forest       |   0.941296 |   0.914737 | {'max_depth': 5, 'n_estimators': 50}        |
| Gradient Boosting   |   0.939946 |   0.910849 | {'learning_rate': 0.01, 'n_estimators': 50} |

| Model                   |   Accuracy |   F1 Score |
|-------------------------|------------|------------|
| Distilbert-base-uncased |  0.0203779 | 0.00410165 |

| Model       Fine Tuned            |   Accuracy |   F1 Score |
|-------------------------|------------|------------|
| distilbert-base-uncased |   0.952497 |   0.943384 |