# **Project Quick Start**

This document contains the essential commands and data context for the TENG Gait Analysis project.

## **1\. Project Data Files**

Your .gitignore file correctly separates your code (in Git) from your data (in Google Drive). The two most important data sources are:

1. **User\_Gait\_Data\_Master/User\_Data\_Labelled/**  
   * **Contains:** 84 CSV files, one for each trial.  
   * **Columns:** Time(s), Voltage(V)  
   * **Purpose:** Used for the 4-class **Activity Classification** (stand, walk, run, jump).  
2. **User\_Gait\_Data\_Master/QOM/gait\_labels\_qom.csv**  
   * **Contains:** The master label file.  
   * **Columns:** subject\_id, height\_cm, weight\_kg, gender, activity, trial, qom, file\_path.  
   * **Purpose:** This is the "ground truth" file used for all other models:  
     * **QOM Classification:** Uses the qom column.  
     * **Bio-authentication:** Uses the weight\_kg column.  
     * **BMI Calculation:** Can be calculated from height\_cm and weight\_kg.

## **2\. Local Setup (VScode, etc.)**

### **Activate Environment**

*Run this first every time you work.*

**PowerShell (Windows):**

.\\venv\\Scripts\\Activate.ps1

**Command Prompt (Windows):**

.\\venv\\Scripts\\activate

**macOS / Linux:**

source venv/bin/activate

### **Check Data Integrity**

*Run this script to make sure all your data files are present and have the correct columns before you train.*

python validate\_data.py

### **Run Machine Learning**

*This is the main analysis notebook. Run it with Jupyter.*

jupyter notebook gait\_analysis\_ml.ipynb

## **3\. Google Colab Setup (Beginner Friendly)**

This is the easiest way to get started with training.

**Step 1: Upload Your Data to Google Drive**

* Upload your single .zip file (User\_Gait\_Data\_Master.zip) to the *main* folder of your Google Drive. Do **not** unzip it there.

**Step 2: Open Colab and Upload Notebook**

* Go to [colab.research.google.com](https://colab.research.google.com).  
* Click **File \-\> Upload notebook...**  
* Upload the gait\_analysis\_ml.ipynb file from your project.

**Step 3: Run the First Cells**

* Once the notebook is open, just run the cells from the top.  
* The **first cell** will ask you to connect to your Google Drive. Click the link, get the authorization code, and paste it in.  
* The **second cell** will unzip your data *inside* the Colab environment (this is fast).

After that, you are ready to run all the machine learning models directly in your browser\!

## **4\. Git Workflow**

**A. Get team updates (Do this before you start working):**

git pull

**B. Save and share your work (Do this when you are done):**

\# 1\. Add all your changed files  
git add .

\# 2\. Save your changes with a message  
git commit \-m "Your update message (e.g., 'updated xgb model')"

\# 3\. Push your changes to GitHub  
git push  
