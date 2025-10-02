Anomalies in ICU Monitoring
1. Vital Signs Anomalies
2. Heart rate (HR): tachycardia, bradycardia, sudden fluctuations.
3. Blood pressure (BP): hypotension, hypertension spikes, abnormal variability.
4. Respiratory rate (RR): apnea events (no breaths detected), tachypnea, erratic patterns.
5. Oxygen saturation (SpO₂): hypoxemia, sudden desaturations, sensor dropouts.
6. Temperature: hyperpyrexia, hypothermi.

Autoencoder:-
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
