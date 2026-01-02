import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load saved artifacts
# -------------------------------------------------
model = joblib.load("insurance_claim_xgboost_model.pkl")
preprocess = joblib.load("insurance_claim_preprocess.pkl")
threshold = joblib.load("insurance_claim_threshold.pkl")

# -------------------------------------------------
# Streamlit Config
# -------------------------------------------------
st.set_page_config(page_title="Car Insurance Claim Prediction", layout="centered")

st.title("🚗 Car Insurance Claim Prediction")
st.write("Predict the probability that a customer will file an insurance claim.")

st.divider()

# -------------------------------------------------
# USER INPUTS
# -------------------------------------------------
st.subheader("Policy & Customer Details")

policy_tenure = st.number_input("Policy Tenure", 0.0, 2.0, 0.5)
age_of_car = st.number_input("Age of Car (normalized)", 0.0, 1.0, 0.2)
age_of_policyholder = st.number_input("Age of Policyholder (normalized)", 0.0, 1.0, 0.4)
population_density = st.number_input("Population Density", 0, 8000)

st.subheader("Vehicle Details")

make = st.selectbox("Car Make (Encoded)", [1, 2, 3, 4, 5])
model_name = st.selectbox("Car Model (Encoded)", [1, 2, 3, 4, 5])
segment = st.selectbox("Car Segment", ["A", "B1", "B2", "C1", "C2"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])
engine_type = st.selectbox("Engine Type", ["MPFI", "CRDI", "VVT"])
rear_brakes_type = st.selectbox("Rear Brakes Type", ["Drum", "Disc"])
steering_type = st.selectbox("Steering Type", ["Power", "Manual"])

airbags = st.slider("Number of Airbags", 1, 6, 2)
ncap_rating = st.slider("NCAP Safety Rating", 0, 5, 3)

st.subheader("Engine & Dimensions")

max_power = st.number_input("Max Power (bhp)", 50.0, 200.0, 90.0)
displacement = st.number_input("Engine Displacement (cc)", 500, 2000, 1200)
gross_weight = st.number_input("Gross Weight (kg)", 800, 2000, 1300)
turning_radius = st.number_input("Turning Radius (m)", 4.0, 6.0, 5.0)

length = st.number_input("Car Length (mm)", 3000, 5000, 3800)
width = st.number_input("Car Width (mm)", 1400, 2000, 1700)
height = st.number_input("Car Height (mm)", 1400, 2000, 1550)

# -------------------------------------------------
# BUILD INPUT DATAFRAME (MATCH TRAINING SCHEMA)
# -------------------------------------------------
input_df = pd.DataFrame([{
    "policy_tenure": policy_tenure,
    "age_of_car": age_of_car,
    "age_of_policyholder": age_of_policyholder,
    "population_density": population_density,
    "make": make,
    "segment": segment,
    "model": model_name,
    "fuel_type": fuel_type,
    "max_power": max_power,
    "engine_type": engine_type,
    "airbags": airbags,
    "rear_brakes_type": rear_brakes_type,
    "displacement": displacement,
    "cylinder": 4,           # fixed default (training value)
    "transmission_type": transmission_type,
    "gear_box": 5,           # fixed default
    "steering_type": steering_type,
    "turning_radius": turning_radius,
    "length": length,
    "width": width,
    "height": height,
    "gross_weight": gross_weight,
    "ncap_rating": ncap_rating
}])

# -------------------------------------------------
# FORCE DATA TYPES (🔥 CRITICAL FIX 🔥)
# -------------------------------------------------

# ALL categorical → STRING
categorical_cols = [
    "make",
    "model",
    "segment",
    "fuel_type",
    "transmission_type",
    "engine_type",
    "rear_brakes_type",
    "steering_type"
]

for col in categorical_cols:
    input_df[col] = input_df[col].astype(str)

# ALL numeric → NUMERIC
numeric_cols = [
    "policy_tenure",
    "age_of_car",
    "age_of_policyholder",
    "population_density",
    "airbags",
    "max_power",
    "displacement",
    "gross_weight",
    "turning_radius",
    "length",
    "width",
    "height",
    "ncap_rating",
    "cylinder",
    "gear_box"
]

for col in numeric_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if st.button("Predict Claim Probability"):
    try:
        processed_input = preprocess.transform(input_df)
        probability = model.predict_proba(processed_input)[0][1]

        st.subheader("Prediction Result")
        st.write(f"📊 **Claim Probability:** `{probability:.3f}`")

        if probability >= threshold:
            st.error("⚠️ High Risk: Claim Likely")
        else:
            st.success("✅ Low Risk: Claim Unlikely")

    except Exception as e:
        st.error("Prediction failed due to preprocessing mismatch.")
        st.exception(e)

st.divider()
st.caption("Model: Tuned XGBoost | Threshold Optimized | SHAP Explained")
