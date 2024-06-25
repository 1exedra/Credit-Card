# Fraud-Detection

Interpreting the classification report

The classification report provides detailed metrics for each class:

Precision: Precision is the ratio of correctly predicted positive observations to the total predicted positives
Precision = TP / (TP + FP)

For class 0: 1.00 100% of the transactions predicted as non-fraudulent are actually non-fraudulent

For class 1: 0.97 97% of the transactions predicted as fraudulent are actually fraudulent

Recall: Recall is the ratio of correctly predicted positive observations to all observations in the actual class
Recall = TP / (TP + FN)

For class 0: 1.00 100% of the actual non-fraudulent transactions are correctly predicted
For class 1: 0.80 80% of the actual fraudulent transactions are correctly predicted
F1-Score: The F1 score is the weighted average of Precision and Recall
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

For class 0: 1.00
For class 1: 0.88
Support: The number of actual occurrences of the class in the test data

For class 0: 19954
For class 1: 46
Accuracy: The ratio of correctly predicted observations to the total observations

Overall accuracy: 1.00 100%
Macro Average: The unweighted mean of precision, recall, and F1-score for each class

Precision: 0.99
Recall: 0.90
F1-Score: 0.94
Weighted Average: The weighted mean of precision, recall, and F1-score for each class, taking into account the support number of true instances for each label

Precision: 1.00
Recall: 1.00
F1-Score: 1.00
ROC AUC Score
The ROC AUC (Receiver Operating Characteristic Area Under Curve) score is a measure of the model's ability to distinguish between classes.

ROC AUC Score: 0.954987776233162

This score ranges from 0 to 1, where 1 indicates perfect prediction and 0.5 indicates a random guess. A score of approximately 0.95 means that the model has a high capability of distinguishing between fraudulent and non-fraudulent transactions

Class 0 (Non-Fraudulent Transactions): The model performs exceptionally well with perfect precision, recall, and F1-score.
Class 1 (Fraudulent Transactions): The model has high precision (97%) but slightly lower recall (80%). This means while the model is good at identifying fraudulent transactions when it predicts them, it misses about 20% of the actual fraudulent transactions
Overall Performance: The model achieves very high overall accuracy and an excellent ROC AUC score, indicating strong performance in distinguishing between fraudulent and non-fraudulent transactions
