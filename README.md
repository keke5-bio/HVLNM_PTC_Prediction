# HVLNM_PTC_Prediction
A CatBoost-based machine learning system for predicting Hv lymph node metastasis in thyroid cancer patients.
\# Thyroid Cancer Lymph Node Metastasis Prediction System
\## 📌 Overview
This project implements a machine learning-based prediction system for assessing lymph node metastasis risk in thyroid cancer patients. The system uses CatBoost algorithm trained on clinical features and provides an interactive web interface via Streamlit.
\## 🎯 Key Features

\- \*\*CatBoost Model\*\*: Advanced gradient boosting algorithm for accurate prediction

\- \*\*Streamlit Web Interface\*\*: User-friendly interactive application

\- \*\*Feature Importance Analysis\*\*: Visual explanation of model decisions

\- \*\*Clinical Integration\*\*: Combines ultrasound, pathological, and laboratory indicators

\## 📊 Model Performance

\- \*\*Test Set AUC\*\*: 0.7496

\- \*\*Test Set Accuracy\*\*: 0.7285

\- \*\*Best Iteration\*\*: 11

\- \*\*Key Features\*\*: Tumor Size, Color Doppler Flow, New Lesion Type

\## 🚀 Quick Start

\### Prerequisites

\- Python 3.9+

\- Git

\### Installation

```bash

\# Clone the repository

git clone https://github.com/keke5-bio/HVLNM_PTC_Prediction.git

cd HVLNM_PTC_Prediction

\# Install dependencies

pip install -r requirements.txt

\# Train the model (optional, pre-trained model included)

python train\_model.py


\# Launch the web application

streamlit run app.py
