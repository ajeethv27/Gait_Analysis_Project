Project Summary: Gait Analysis using a TENG Sensor

This is a Master's thesis project that uses machine learning to analyze human movement (gait) from a custom-built sensor.

The Sensor:
The data is collected from a single, low-cost TENG (Triboelectric Nanogenerator) sensor, which is placed in a shoe insole. This sensor generates a unique voltage signal based on the pressure and movement of a person's foot.

The Data:
The dataset consists of 84 recordings from 12 different people performing four activities: standing, walking, running, and jumping.

The Primary Goals:
The project has four main goals for the machine learning model:

Activity Classification & Metrics: To accurately identify what the person is doing (e.g., standing, walking, running) and extract key metrics like activity duration and step count from the signal.

Clinical QOM Assessment: To classify the 1-10 Quality of Movement (QOM) score into three clinical categories: 'Intervention Priority' (1-3), 'Improvement Recommended' (4-6), and 'Healthy Baseline' (7-10). For clinical use, the model will also provide a confidence rating for its prediction.

Bio-authentication (A Novel Feature): To test a creative idea: can the sensor's data be used as a "fingerprint"? The model will try to predict a person's body weight just from their gait pattern, which could be a new way to identify users.

Rehabilitation Focus: To analyze which movement patterns (e.g., in 'run' or 'jump' signals) are linked to the probability of specific muscular weaknesses (like quadriceps), with the future goal of suggesting customized workout plans.

The Setup:
The project is set up for teamwork. All the Python code (.py scripts) is shared and version-controlled using Git (on GitHub). The large data files (the 84 CSVs) are stored separately on Google Drive, and the main README.md file links everything together so the team can collaborate on the analysis.