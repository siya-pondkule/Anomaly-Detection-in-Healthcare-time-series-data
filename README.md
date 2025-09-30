Synopsis: https://1drv.ms/w/c/0c4b1487469d4d26/EV8VT-6lmfFFgMn1XWQ9LMEBnwu5LjH73wRAK4kfeOyTOw?e=CzkIwY

Reviews of Papers: https://1drv.ms/x/c/0c4b1487469d4d26/EfX9_MLUjDxIsO1pw_LwV-UBZ3Ip-eRrJWvlFhACHn2RSA?e=bIy32O


Anomalies in ICU Monitoring
1. Vital Signs Anomalies
2. Heart rate (HR): tachycardia, bradycardia, sudden fluctuations.
3. Blood pressure (BP): hypotension, hypertension spikes, abnormal variability.
4. Respiratory rate (RR): apnea events (no breaths detected), tachypnea, erratic patterns.
5. Oxygen saturation (SpO₂): hypoxemia, sudden desaturations, sensor dropouts.
6. Temperature: hyperpyrexia, hypothermi.

Wearables Dataset
This folder contains the data file “StresSense.csv”.
The dataset comprises a total of 495,446 data points with twelve columns. These columns include a timestamp, user ID, activity label, and three-dimensional values from the accelerometer (A), gyroscope (G), and magnetometer (M), labeled as follows [1].
1. TimeStamp: time stamp of a record with sampling rate of 50 Hz.
2. Gx: angular velocity on the x axis of the gyroscope.
3. Gy: angular velocity on the y axis of the gyroscope.
4. Gz: angular velocity on the z axis of the gyroscope.
5. Ax: acceleration on the x axis of the accelerometer.
6. Ay: acceleration on the y axis of the accelerometer.
7. Az: acceleration on the z axis of the accelerometer.
8. Mx: magnitude on the x axis of the magnetometer.
9. My: magnitude on the y axis of the magnetometer.
10. Mz: magnitude on the z axis of the magnetometer.
11. User: unique identification number for each user.
12. Activity: name of an activity being performed.

| **Type of Anomaly**                | **Examples in Dataset**                                                                                           | **What It Can Predict**                                                                                          |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Sensor-level anomalies**         | - Sudden spikes in accelerometer/gyroscope <br> - Flat-line signals <br> - Missing values                         | - Stress episodes (jerky, abrupt movements) <br> - Boredom (very still signals) <br> - Distraction/device misuse |
| **Activity-label anomalies**       | - Mislabeling (e.g., “eating” but actually face touching) <br> - Overlapping habits (nail-biting + face touching) | - Mixed or unconscious stress behaviors <br> - Higher anxiety if multiple habits overlap                         |
| **User-behavior anomalies**        | - Outlier motion patterns (stronger/weaker hand use) <br> - Variable session lengths (smoking longer/shorter)     | - Personalized stress signatures <br> - Stress intensity estimation (duration of stress behavior)                |
| **Protocol/environment anomalies** | - Movement during “staying still” (fidgeting, finger tapping) <br> - Unnatural behavior due to sensor holder      | - Hidden boredom/stress signals (restlessness during stillness) <br> - Contextual stress influences              |
