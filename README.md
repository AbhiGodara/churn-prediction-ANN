# 🏦 Bank Customer Churn Prediction

A Machine Learning–powered web application that predicts whether a bank customer is likely to **churn (leave the bank)** or **stay**, using an **Artificial Neural Network (ANN)** and deployed with **Streamlit**.

---

## 📌 Table of Contents

- Project Overview
- Live Demo
- Features
- Tech Stack
- Model Information
- Input Feature Columns
- Project Structure
- Installation
- Usage
- Model Artifacts
- Notebooks
- Troubleshooting
- Future Improvements
- License

---

## 📖 Project Overview

Customer churn prediction helps banks identify customers who are likely to leave their services.  
This project uses historical customer data, preprocesses it using encoders and scalers, and applies an ANN model to predict churn probability.

The trained model is deployed using **Streamlit** for real-time predictions.

---

## 🌐 Live Demo

> **Dummy Live App Link**  
🔗 https://abhishek-churn-prediction.streamlit.app/

---

## ✨ Features

- Interactive Streamlit web interface
- Real-time churn prediction
- Probability-based output
- Pre-trained ANN model
- Handles categorical & numerical features
- Clean and modular ML pipeline

---

## 🧰 Tech Stack

- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Pickle

---

## 🧠 Model Information

- **Model Type:** Artificial Neural Network (ANN)
- **Problem Type:** Binary Classification
- **Output:** Churn Probability (0–1)
- **Threshold:**  
  - `>= 0.5` → Customer likely to churn  
  - `< 0.5` → Customer likely to stay

---

## 🔢 Input Feature Columns

### Numerical Features
- `Age`
- `CreditScore`
- `Balance`
- `Tenure`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

### Categorical Features
- `Gender` (Label Encoded)
- `Geography` (One-Hot Encoded into):
  - `Geography_France`
  - `Geography_Germany`
  - `Geography_Spain`

> Feature order consistency is maintained using `column_order.pkl`.

---

## 🗂️ Project Structure

```bash
bank-customer-churn-prediction/
│
├── app.py                     # Streamlit application
├── model.h5                   # Trained ANN model
├── scaler.pkl                 # StandardScaler object
├── onehotencoder.pkl          # OneHotEncoder for Geography
├── label_encoder_gender.pkl   # LabelEncoder for Gender
├── column_order.pkl           # Feature column order
│
├── notebooks/
│   ├── salary_regression.ipynb
│   ├── hyperparametertuning_ann.ipynb
│   └── prediction.ipynb
│
├── requirements.txt
└── README.md
