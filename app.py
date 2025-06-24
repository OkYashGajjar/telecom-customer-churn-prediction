import os

import streamlit as st
import pandas as pd
import joblib

# 1. Load your trained pipeline
@st.cache_resource
def load_model(path='best_churn_model.pkl'):
    return joblib.load(path)

model = load_model()

st.title("📊 Customer Churn Predictor")

st.markdown("""
Fill in the customer details below.  
Hit **Predict** to see if they’re likely to churn.
""")

# 2. Sidebar (or main) inputs
st.sidebar.header("Customer Profile")

tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly = st.sidebar.number_input("MonthlyCharges", min_value=0.0, max_value=500.0, value=70.0)
total = st.sidebar.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=840.0)

partner        = st.sidebar.selectbox("Partner", ['Yes','No'])
dependents     = st.sidebar.selectbox("Dependents", ['Yes','No'])
multilines     = st.sidebar.selectbox("MultipleLines", ['Yes','No','No phone service'])
internet       = st.sidebar.selectbox("InternetService", ['DSL','Fiber optic','No'])
contract       = st.sidebar.selectbox("Contract", ['Month-to-month','One year','Two year'])
payment_method = st.sidebar.selectbox("PaymentMethod", [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
])

# 3. Package into a DataFrame
input_df = pd.DataFrame([{
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'Partner': partner,
    'Dependents': dependents,
    'MultipleLines': multilines,
    'InternetService': internet,
    'Contract': contract,
    'PaymentMethod': payment_method
}])

st.write("**Input data:**")
st.dataframe(input_df)

# 4. Predict button
if st.button("Predict Churn"):
    pred_proba = model.predict_proba(input_df)[0,1]
    pred_class = model.predict(input_df)[0]
    
    st.write(f"**Churn probability:** {pred_proba:.2f}")
    label = "🚨 Will churn" if pred_class==1 else "✅ Will stay"
    st.write(f"**Prediction:** {label}")
