TENG Gait Analysis Project

This repository contains the Python scripts and analysis for a Master's thesis on gait analysis using Triboelectric Nanogenerator (TENG) sensors.

1. Get the Data

IMPORTANT: The dataset is too large for Git and is stored externally.

Download the Data:
Download the complete dataset (User_Gait_Data_Master.zip) from the link below:
https://drive.google.com/file/d/119YLrW2UjjWO8pUIc35z5WU_bZfHetc7/view?usp=sharing
Unzip the Data:
Unzip the file into the root of this project folder. Your directory structure should look like this:

Gait_Analysis_Project/
├── .gitignore
├── requirements.txt
├── gait_analysis.py
└── User_Gait_Data_Master/  <-- The folder you just unzipped
    ├── User_Data_Labelled/
    └── QOM/


2. Setup Environment (One Time)

First, create a clean virtual environment:

python -m venv venv


Next, activate it:

Windows (PowerShell):

.\venv\Scripts\Activate.ps1


macOS / Linux:

source venv/bin/activate


3. Install Libraries (One Time)

With your environment active, install all required libraries:

pip install -r requirements.txt


4. Run Analysis

To run the main analysis script:

(venv) > python gait_analysis.py
