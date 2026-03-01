# Loan Default Prediction System
---

## 1. Project Overview

This repository contains a production ready machine learning solution to predict whether a loan applicant will default.

The project moves from raw structured data to an optimized **XGBoost** model served through a high performance **FastAPI** REST API.

---

## 2. Dataset and Engineering Decisions

The original dataset contained 34 features. The modeling process involved two key refinement phases.

---

### Phase A: Eliminating Data Leakage

During early experiments, training with all features produced unrealistic 99 percent plus accuracy.

**Root cause**

Features such as:

* rate_of_interest
* interest_rate_spread

are determined by the bank after the approval decision is made.

Using them to predict loan status introduces data leakage because the model would be learning from future information.

**Action Taken**

These variables were removed so the model only uses information available at the time of application.

This ensures realistic and production valid performance.

---

### Phase B: Strategic Feature Selection

To reduce overfitting and improve inference speed, the model was restricted to the six most predictive features aligned with the three pillars of lending.

| Feature      | Category    | Why It Matters                        |
| ------------ | ----------- | ------------------------------------- |
| Credit_Score | Character   | Historical reliability indicator      |
| LTV          | Capital     | Loan to Value risk against collateral |
| dtir1        | Capacity    | Debt to Income repayment ability      |
| loan_type    | Condition   | Risk differences across loan products |
| age          | Demographic | Life stage risk pattern               |
| Region       | Economic    | Geographic economic variation         |

This design reduces noise, improves generalization, and lowers API latency.

---

## 3. Technical Implementation

### Preprocessing Pipeline

Built using ColumnTransformer with:

* Median imputation for numerical features
* Most frequent imputation for categorical features
* OneHotEncoding for categorical variables

The preprocessing is embedded inside a single Scikit Learn Pipeline to prevent data leakage during training and inference.

---

### Model Optimization

Hyperparameter tuning was performed using Optuna Bayesian optimization over 30 trials.

Best performing model:

* XGBoost Classifier
* n_estimators: 169
* learning_rate: 0.125
* max_depth: 4

---

### API Architecture

Built using FastAPI and Pydantic.

Features:

* Strict type validation
* Structured request schema
* Custom error handling
* Clear JSON response format

The deployed model is a serialized Scikit Learn pipeline containing both preprocessing and classifier logic.

---

## 4. Performance Metrics

* Final Test Accuracy: 87.32 percent
* Inference Speed: less than 20 milliseconds per request
* Stable handling of invalid inputs without crashing

## Model Performance Comparison

| Model                | Best Parameters                                                   | Test Accuracy |
|----------------------|------------------------------------------------------------------|---------------|
| Logistic Regression  | C = 10.0, solver = lbfgs                                         | 0.5916        |
| Random Forest        | max_depth = 10, n_estimators = 100                               | 0.8497        |
| XGBoost (Winner)     | learning_rate = 0.1, max_depth = 6, n_estimators = 150          | 0.8728        |

---

## 5. Project Structure

```
├── main.py
├── train.py
├── best_model.pkl
├── requirements.txt
├── Loan_Default.csv
├── api_screenshots/
└── README.md
```

---

## 6. Setup and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run the API

```bash
uvicorn main:app --reload
```

After starting the server, open:

```
http://127.0.0.1:8000/docs
```

to interact with the API using the Swagger interface.

