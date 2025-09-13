# Anomaly-Detection-in-Financial-Transactions-using-Autoencoders-Clustering
 Project Overview:

This project focuses on fraud detection in financial transactions using a combination of deep learning (Autoencoders) and unsupervised clustering (OPTICS/DBSCAN).
The hybrid approach is designed to maximize the detection of fraudulent transactions by leveraging both reconstruction error thresholds and clustering-based anomaly detection.


Steps in the Project

Step 1: Data Loading and Preprocessing

Loaded financial transaction dataset.

Standardized/normalized features for neural network training.

Split dataset into train/test sets ensuring fraud class distribution is preserved.

Step 2: Train Autoencoder Model

Built and trained a deep autoencoder on the majority (non-fraud) transactions.

Used MSE reconstruction error as the anomaly score.

Step 3: Evaluate Autoencoder Reconstruction Errors

Computed reconstruction error for each transaction.

Analyzed error distribution between fraud and non-fraud classes.

Applied threshold-based detection using Precision-Recall Curve and ROC Curve.

Step 4: Threshold Selection (PR Curve & ROC Curve)

Identified optimal thresholds using:

Youden’s J statistic (from ROC).

Best trade-off point (from PR Curve).

Generated predictions:

Transactions with error ≥ threshold → Fraud (1).

Transactions with error < threshold → Normal (0).

Step 5: Run Clustering (OPTICS) on Reconstruction Errors

Applied OPTICS clustering on reconstruction errors.

Grouped transactions into normal vs anomalous clusters.

Obtained anomaly predictions independent of Autoencoder.

Step 6: Hybrid Decision Rule

Combined predictions from Autoencoder + OPTICS:

If Autoencoder predicts fraud OR OPTICS flags anomaly above cutoff percentile, label as fraud.

Evaluated performance across multiple cutoff percentiles (95th, 97th, 98th, 99th, 99.5th).

Final hybrid model achieved the highest recall for fraud detection.

Step 7: Model Comparison

Compared three approaches:

Autoencoder-only

OPTICS-only

Hybrid Model

Reported metrics: Precision, Recall, F1-score, Confusion Matrix

✅ Conclusion

Since the aim of fraud detection is to capture as many fraudulent transactions as possible, recall is the most critical metric.

The Autoencoder-only model achieved good precision but missed some fraud cases.

The OPTICS-only model identified additional anomalies but with lower precision.

The Hybrid Model achieved the highest recall for fraud detection, making it the best choice in scenarios where minimizing false negatives is more important than minimizing false positives.

Thus, the Hybrid Autoencoder + OPTICS approach is a strong candidate for real-world fraud detection tasks.

📊 Tech Stack

Python

Pytorch (Autoencoder)

Scikit-learn (ROC, PR curves, metrics, clustering)

Matplotlib, Seaborn (Visualization)
