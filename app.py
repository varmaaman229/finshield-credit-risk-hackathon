import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the model and data
model = joblib.load('models/lightgbm_model.pkl')
test_df = pd.read_csv('data/processed/test.csv')
features = [col for col in test_df.columns if col not in ['SK_ID_CURR', 'TARGET']]
X_test = test_df[features]

st.title("Credit Risk Prediction and Explanation")

# Sidebar for user input
st.sidebar.header("Select an Applicant")
applicant_id = st.sidebar.selectbox("Applicant ID", test_df['SK_ID_CURR'].unique())

# Get the selected applicant's data
applicant_data = X_test[test_df['SK_ID_CURR'] == applicant_id]

# Make a prediction
prediction = model.predict_proba(applicant_data)[:, 1][0]

st.header("Prediction")
st.write(f"The probability of default for applicant {applicant_id} is: **{prediction:.2f}**")

# Explain the prediction
st.header("Prediction Explanation")
explainer = shap.TreeExplainer(model)
shap_values_single = explainer.shap_values(applicant_data)

# Create and display the first plot, then close it
fig1, ax1 = plt.subplots()
shap.force_plot(explainer.expected_value, shap_values_single, applicant_data, matplotlib=True, show=False)
st.pyplot(fig1, bbox_inches='tight')
plt.close(fig1)


st.write("---")

st.header("Feature Importance")
shap_values_all = explainer.shap_values(X_test)

# Create and display the second plot, then close it
fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values_all, X_test, plot_type="bar", show=False)
st.pyplot(fig2, bbox_inches='tight')
plt.close(fig2)
