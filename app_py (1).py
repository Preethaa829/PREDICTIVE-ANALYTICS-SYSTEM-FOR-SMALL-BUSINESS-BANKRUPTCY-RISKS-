import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---- Load Model & SHAP CSV ----
rf_model = joblib.load('rf_model.pkl')  # Load pre-trained model (Random Forest model)
shap_df = pd.read_csv('top_shap_features.csv')  # Load SHAP features (or importance) data

# ---- Predictor Columns (Match model training) ----
predictors = [
    'Asset_Turnover_ratio', 'ROA', 'Interest_on_Debt', 'Debt_to_Equity',
    'WC_to_Sales', 'Quick_ratio', 'Operating_Margin',
    'Net_Margin', 'Profit_Margin', 'Operating_CF',
    'Liabilities_to_Assets', 'Return_on_Assets', 'NetWC_to_Assets',
    'EBIT_to_Assets', 'EBIT_to_Sales', 'Current_Ratio', 'Cash_Ratio',
    'Working_Capital_Ratio', 'Equity_to_Liabilities', 'Net_Sales_to_Assets',
    'Net_Worth_to_Debt', 'CF_to_debt', 'CF_to_sales', 'CF_coverage',
    'CF_to_equity', 'CF_to_Assets', 'Gross_Margin', 'RE_to_Sales',
    'RE_to_Assets', 'RE_to_NetIncome', 'Inventory_Turnover', 'Inventory_to_Sales'
]

# ---- Sample Input Options ----
sample_inputs = {
    "Manual Input": {k: "" for k in predictors},
    
    "Non-Bankrupt Example": {
        'Asset_Turnover_ratio': 2.0, 'ROA': 0.18, 'Interest_on_Debt': 0.01, 'Debt_to_Equity': 0.25,
        'WC_to_Sales': 0.22, 'Quick_ratio': 2.8, 'Operating_Margin': 0.22, 'Net_Margin': 0.20,
        'Profit_Margin': 0.21, 'Operating_CF': 3500000, 'Liabilities_to_Assets': 0.35,
        'Return_on_Assets': 0.18, 'NetWC_to_Assets': 0.24, 'EBIT_to_Assets': 0.19,
        'EBIT_to_Sales': 0.25, 'Current_Ratio': 3.0, 'Cash_Ratio': 1.5, 'Working_Capital_Ratio': 2.8,
        'Equity_to_Liabilities': 2.2, 'Net_Sales_to_Assets': 2.1, 'Net_Worth_to_Debt': 1.8,
        'CF_to_debt': 0.20, 'CF_to_sales': 0.22, 'CF_coverage': 2.2, 'CF_to_equity': 0.20,
        'CF_to_Assets': 0.18, 'Gross_Margin': 0.38, 'RE_to_Sales': 0.26, 'RE_to_Assets': 0.22,
        'RE_to_NetIncome': 1.2, 'Inventory_Turnover': 7.5, 'Inventory_to_Sales': 0.09
    },
    
    "Bankrupt Example": {
        'Asset_Turnover_ratio': 0.7,    # Lower asset turnover reflects inefficiency
        'ROA': -0.05,                   # Negative ROA indicates losses
        'Interest_on_Debt': 0.08,       # High interest on debt reflects difficulty in managing debt
        'Debt_to_Equity': 1.2,          # High debt-to-equity ratio signifies financial distress
        'WC_to_Sales': 0.05,            # Poor working capital management
        'Quick_ratio': 0.8,             # Quick ratio below 1 indicates liquidity issues
        'Operating_Margin': -0.12,      # Negative operating margin indicates operational inefficiencies
        'Net_Margin': -0.15,            # Negative net margin indicates the company is losing money
        'Profit_Margin': -0.1,          # Negative profit margin shows overall inefficiency
        'Operating_CF': 200000,         # Low operating cash flow signals liquidity issues
        'Liabilities_to_Assets': 0.85,  # High liabilities compared to assets suggests financial instability
        'Return_on_Assets': -0.07,      # Negative return on assets indicates poor asset utilization
        'NetWC_to_Assets': 0.05,        # Low working capital to assets ratio suggests financial struggles
        'EBIT_to_Assets': -0.06,        # Negative EBIT to assets ratio signals poor performance
        'EBIT_to_Sales': -0.08,         # Negative EBIT to sales indicates unprofitable sales
        'Current_Ratio': 0.9,           # Low current ratio suggests potential liquidity problems
        'Cash_Ratio': 0.3,              # Very low cash ratio indicates cash shortage
        'Working_Capital_Ratio': 0.12,  # Low working capital ratio reflects financial stress
        'Equity_to_Liabilities': 0.5,   # Low equity-to-liabilities ratio indicates financial instability
        'Net_Sales_to_Assets': 1.0,     # Low sales-to-assets ratio indicates poor utilization of assets
        'Net_Worth_to_Debt': 0.3,       # Low net worth to debt ratio indicates high debt burden
        'CF_to_debt': 0.05,             # Low cash flow to debt ratio indicates difficulty in covering debt
        'CF_to_sales': 0.05,            # Low cash flow to sales indicates poor liquidity management
        'CF_coverage': 0.6,             # Low cash flow coverage ratio
        'CF_to_equity': 0.07,           # Very low cash flow to equity indicates liquidity issues
        'CF_to_Assets': 0.05,           # Very low cash flow to assets ratio indicates poor financial health
        'Gross_Margin': 0.15,           # Low gross margin suggests that the company is struggling to maintain profits
        'RE_to_Sales': -0.02,           # Negative retained earnings to sales ratio indicates poor retention
        'RE_to_Assets': -0.05,          # Negative retained earnings to assets ratio shows a lack of retained earnings
        'RE_to_NetIncome': -0.08,       # Negative retained earnings to net income ratio indicates poor financial health
        'Inventory_Turnover': 4.0,      # Low inventory turnover suggests poor sales and overstocking
        'Inventory_to_Sales': 0.25      # High inventory-to-sales ratio suggests inventory management issues
    }
}

# ---- Streamlit UI ----
st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")
st.title("🏦 Bankruptcy Prediction App with Explainable AI")

# ---- Dropdown to choose input ----
preset = st.selectbox("Choose Sample Input or Manual Entry", list(sample_inputs.keys()))

# ---- Input Fields ----
st.subheader("Enter or Review Financial Details")
user_input = {}

col1, col2 = st.columns(2)
for i, feature in enumerate(predictors):
    default_val = sample_inputs[preset][feature]
    with (col1 if i < 16 else col2):
        user_input[feature] = st.text_input(feature, value=str(default_val), key=feature)

# ---- Prediction ----
if st.button("🔍 Predict Bankruptcy Status"):
    try:
        # Convert input to dataframe
        input_values = [float(user_input[feature]) for feature in predictors]
        input_df = pd.DataFrame([input_values], columns=predictors)

        # Use your trained Random Forest model to predict
        prediction_proba = rf_model.predict_proba(input_df)[0][1]  # Probability of bankruptcy
        threshold = 0.25  # Adjusted threshold to make it more sensitive to bankruptcy
        prediction = 1 if prediction_proba > threshold else 0

        # Show Prediction
        st.markdown("## 🔎 Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ The company is likely to go **BANKRUPT**. (Confidence: {prediction_proba:.2f})")
        else:
            st.success(f"✅ The company is **NOT likely to go bankrupt**. (Confidence: {prediction_proba:.2f})")

        # Show Top SHAP Features
        st.markdown("## 📊 Top Influential Features (SHAP)")
        st.dataframe(shap_df)

    except ValueError:
        st.warning("Please enter valid numeric values for all fields.")
