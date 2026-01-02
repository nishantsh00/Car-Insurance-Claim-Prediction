# 🚗 Car Insurance Claim Prediction

## 📌 Project Overview
This project aims to predict whether a car insurance policyholder is likely to
make an insurance claim using machine learning techniques. The solution is built
as an end-to-end pipeline covering data preprocessing, exploratory data analysis,
modeling, explainability, and deployment.

The final model is deployed using a Streamlit web application for real-time
prediction.

---

## 🎯 Problem Statement
Insurance companies face significant financial risk due to unexpected claims.
Accurately identifying high-risk policyholders helps insurers:
- Improve pricing strategies
- Reduce claim losses
- Optimize underwriting decisions

This project predicts the probability of an insurance claim based on customer,
vehicle, and policy attributes.

---

## 📊 Dataset Description
The dataset contains information related to:
- Policy details (policy tenure)
- Customer demographics (age of policyholder, population density)
- Vehicle specifications (engine details, safety features, dimensions)
- Target variable: `is_claim` (0 = No Claim, 1 = Claim)

The dataset is highly imbalanced, with only ~6% claim cases.

---

## 🔍 Exploratory Data Analysis (EDA)
Key observations from EDA:
- Strong class imbalance in the target variable
- Vehicle age, safety ratings (NCAP), and population density influence claim risk
- Safety features such as airbags reduce claim probability
- Urban areas show higher claim frequency

EDA included:
- Distribution plots
- Target vs feature analysis
- Correlation analysis
- Business-driven insights

---

## ⚙️ Data Preprocessing
Steps performed:
- Cleaned string-based numeric columns (`max_power`, `max_torque`)
- Encoded categorical variables using OneHotEncoder
- Dropped identifier columns
- Applied SMOTE to handle class imbalance
- Train-test split with stratification
- Ensured strict feature consistency for deployment

---

## 🤖 Modeling Approach
Models trained and evaluated:
- Logistic Regression (baseline)
- Decision Tree (baseline)
- Random Forest
- XGBoost (final model)

### Final Model: **XGBoost**
Reasons for selection:
- Best ROC-AUC score on test data
- Better recall for minority (claim) class
- Robust handling of imbalanced data
- Strong performance after hyperparameter tuning

---

## 🎛 Threshold Tuning
Due to class imbalance, the default probability threshold (0.5) was not optimal.

- Multiple thresholds were evaluated
- Threshold = **0.40** provided the best trade-off between precision and recall
- F1-score for the minority class improved

---

## 🔧 Hyperparameter Tuning
- RandomizedSearchCV was used for tuning XGBoost
- Optimized parameters included:
  - Number of trees
  - Tree depth
  - Learning rate
  - Subsampling ratios
- Resulted in improved ROC-AUC on the test set

---

## 🔍 Model Explainability (SHAP)
SHAP (SHapley Additive exPlanations) was used to interpret the final model.

Insights:
- Vehicle age and safety ratings strongly influence predictions
- Population density increases claim probability
- Engine and weight-related features contribute to risk assessment

SHAP improves transparency and trust in the model.

---

## 💾 Model Persistence
The following artifacts were saved using `joblib`:
- Trained XGBoost model
- Preprocessing pipeline
- Optimized decision threshold

These allow inference without retraining.

---

## 🚀 Deployment
The model is deployed using **Streamlit**.

### Features of the App:
- User-friendly interface
- Accepts customer and vehicle inputs
- Predicts claim probability
- Displays risk classification (High / Low)

### Run the App:
```bash
pip install -r requirements.txt
streamlit run app.py
