# Bank Fraud Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Framework-green.svg)](https://xgboost.readthedocs.io/)

An end-to-end data science and machine learning pipeline designed to detect fraudulent financial transactions. This repository demonstrates how advanced analytics, feature engineering, and class-imbalance techniques can be leveraged to mitigate monetary risk and secure banking ecosystems.

---

## 📌 Project Overview

Financial fraud costs institutions billions of dollars annually. This project implements a robust binary classification framework to identify suspicious transactions out of highly imbalanced banking data. 

By analyzing transactional behaviors, account balances, and historical flags, the machine learning models built here optimize the trade-off between capturing maximum fraud (Recall) and minimizing disruptions to legitimate customers (Precision).

### 🔍 Key Implementation Focus Areas
* **Imbalanced Data Management:** Handling extreme class distribution skewness (where fraud accounts for a fraction of a percent of total data).
* **Financial Feature Engineering:** Crafting balance-delta indicators and behavioral flags to expose anomalies.
* **Threshold Optimization:** Moving beyond default classification thresholds to map model predictions directly to business risk metrics.

---

## 🧠 Core Pipeline & Workflow

### 1. Exploratory Data Analysis (EDA)
* Dissected transaction types, volume distributions, and temporal patterns.
* Analyzed the statistical correlation between sudden drops in destination account balances and confirmed fraudulent activities.

### 2. Feature Engineering
* Calculated precise balance discrepancies: `NewBalance - OldBalance`.
* Generated behavioral flags focusing on high-amount thresholds and rapid-succession transfer types.

### 3. Model Architecture & Training
Compared three distinct algorithmic approaches to discover the optimal decision boundary:
* **Logistic Regression:** Established a baseline linear model.
* **Random Forest Classifier:** Captured non-linear interactions via ensemble tree structures.
* **XGBoost (Extreme Gradient Boosting):** Fine-tuned to maximize gradient descent efficiency on complex tabular data.

### 4. Evaluation Metrics
Standard accuracy is highly misleading in fraud analytics. Instead, this system evaluates performance using:
* **Precision-Recall AUC (PR-AUC)**
* **ROC-AUC**
* **Confusion Matrix Profiling**
* **Fraud Recall / Sensitivity**

---

## 🧩 Tech Stack & Dependencies

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Model Serialization:** Joblib

---

## 📂 Repository Structure

```text
├── Analysis.ipynb            # Jupyter Notebook containing full EDA, training, and evaluation
├── Assigment Task.pdf        # Contextual dataset documentation and constraints
├── best_fraud_model_tuned.pkl # Serialized, production-ready XGBoost/Random Forest model
└── README.md                 # Project documentation
