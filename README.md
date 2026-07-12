# 🪖 Smart Airbag Helmet — Pre-Impact Rider State Classifier

> A machine learning pipeline that classifies motorcycle rider states from IMU (MPU6050) data to enable intelligent airbag deployment before impact.

---

## 🎯 Problem Statement

Traditional airbags deploy *reactively* (after impact). This project builds a **pre-impact classifier** that predicts dangerous states in real-time from helmet-mounted IMU sensor data, enabling proactive airbag deployment.

**Not marketed as: "Accident Detection"**
**Marketed as: Pre-Impact Rider State Classification**

---

## 🏷️ Classification Labels

| Label | ID | Description |
|---|---|---|
| 🟢 Normal Riding | 0 | Stable riding, no abnormality |
| 🟡 Pothole / Road Bump | 1 | Short vertical spike, recovers quickly |
| 🟠 Sudden Braking | 2 | Forward deceleration spike |
| 🔴 Crash / Fall | 3 | Multi-axis spike + sustained large deviation |

---

## 🗂️ Project Structure

SmartHelmet/
├── data/
│   ├── synthetic/
│   └── processed/
├── notebooks/
│   ├── 01_generate_data.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_train_model.ipynb
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── models/
├── app/
│   └── dashboard.py
└── README.md

---

## 🛠️ Tech Stack

Python 3.11+ | NumPy | Pandas | Matplotlib | Seaborn | Scikit-Learn | XGBoost | Joblib | Jupyter

---

## 📊 ML Pipeline

Synthetic Time-Series IMU Data (500 sessions x 100 timestamps)
        ↓ EDA
        ↓ Sliding Window (window=20, stride=10)
        ↓ Feature Extraction (mean, std, max, min, magnitude, jerk, tilt)
        ↓ Model Training (RF, XGBoost, SVM, KNN)
        ↓ Evaluation (Accuracy, F1, Confusion Matrix)
        ↓ Live Prediction Pipeline
        ↓ Hardware (ESP32 + MPU6050)
