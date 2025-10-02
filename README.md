Autoencoder
An autoencoder is a type of neural network used for unsupervised learning. It tries to reconstruct its input.

Training:
The autoencoder learns the normal patterns in the data.
It can reconstruct normal data accurately.
If a data point is very different (anomaly), reconstruction error will be high.

Purpose:
Automatic anomaly detection: Detect unusual data without explicit rules.
Noise reduction: Autoencoder smooths out minor anomalies by learning normal patterns.
Future readiness: Once trained, the model can predict anomalies in new ICU data without manually setting thresholds.
Unsupervised learning: Works even when you don’t have labeled anomalies, which is typical in ICU datasets.
