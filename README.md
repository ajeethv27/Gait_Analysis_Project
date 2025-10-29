TENG Gait Analysis Project

This repository contains the Python scripts and analysis for a Master's thesis on gait analysis using Triboelectric Nanogenerator (TENG) sensors.

1. Get the Data

IMPORTANT: The dataset is too large for Git and is stored externally.

Download the Data:
Download the complete dataset (User\_Gait\_Data\_Master.zip) from the link below:
https://drive.google.com/file/d/1vuGJWGcbMUhej5Bq59-X5XrQ-RPkV-gf/view?usp=drive\_linkUnzip the Data:
Unzip the file into the root of this project folder. Your directory structure should look like this:

Gait\_Analysis\_Project/
├── .gitignore
├── requirements.txt
├── gait\_analysis.py
└── User\_Gait\_Data\_Master/  <-- The folder you just unzipped
├── User\_Data\_Labelled/
└── QOM/



2. Setup Environment (One Time)

First, create a clean virtual environment:

python -m venv venv



Next, activate it:

Windows (PowerShell):

.\\venv\\Scripts\\Activate.ps1



macOS / Linux:

source venv/bin/activate



3. Install Libraries (One Time)

With your environment active, install all required libraries:

pip install -r requirements.txt



4. Run Analysis

To run the main analysis script:

(venv) > python gait\_analysis.py

5.(Optionl) My Colab shared workplace
https://colab.research.google.com/drive/1GFmrWjqkLsSQK7itQDQa8DPyf8beq1l7?usp=sharing

