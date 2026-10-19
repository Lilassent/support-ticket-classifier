# Support Ticket Classifier

A text classification project that assigns banking support tickets to one of 77 intent categories.

## Dataset

The project uses the BANKING77 dataset.

- Training examples: 10,003
- Test examples: 3,080
- Categories: 77

## Model Comparison

The models were compared using the same validation split and TF-IDF features.

| Model | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|
| Linear SVM | 0.871 | 0.871 |
| RBF SVM | 0.863 | 0.864 |
| Logistic Regression | 0.845 | 0.842 |
| Random Forest | 0.835 | 0.834 |
| Multinomial Naive Bayes | 0.808 | 0.781 |
| K-Nearest Neighbors | 0.731 | 0.729 |
| Decision Tree | 0.719 | 0.720 |

Linear SVM was selected because it achieved the highest validation accuracy and macro-F1 score.

## Test Results

The final Linear SVM achieved:

- Accuracy: 0.893
- Macro-F1: 0.894

## Make a Prediction

The saved model and TF-IDF vectorizer are stored in `models/support_ticket_classifier_77.joblib`.

```python
import joblib

artifact = joblib.load("models/support_ticket_classifier_77.joblib")

ticket = "I transferred money, but the recipient has not received it."
ticket_vector = artifact["vectorizer"].transform([ticket])
prediction = artifact["model"].predict(ticket_vector)[0]

print(prediction)