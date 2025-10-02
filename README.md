Research Paper on ICU Monitoring - https://journals.sagepub.com/doi/full/10.1177/1460458219850323

Anomalies Detected in paper:
1. Old dataset (MIMIC-II only): Results may not generalize since no external validation was done.
2. SMOTE issue: Oversampling may have been applied before cross-validation → risk of data leakage.
3. Unfair comparison: ML at 6 hours was compared to APACHE/SAPS/SOFA at 24–48 hours (not same time window).
4. Limited metrics: Only AUROC is shown; no calibration, sensitivity/specificity, or threshold-based results.
5. Imputation choices: Used simple mean/EM imputation, ignoring more advanced time-series handling.
6. Feature details missing: Little info on how time-series features were engineered.
7. No clinical validation: No prospective testing or discussion of real-world deployment risks.

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

Strengths of my approach vs. the research paper which i refer
1. No SMOTE/data leakage risk.
2. Richer anomaly handling (threshold-based + autoencoder reconstruction errors).
3. Multiple metrics (MSE, R², anomaly %), not just AUROC.
4. More robust missing-data filling pipeline.
5. correct anomalies and save “cleaned” datasets (they only predicted mortality).
