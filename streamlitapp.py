import os

# Try installing joblib at runtime
os.system('pip install joblib')
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='streamlit')

# Define the feature names used during training
feature_names = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
    'Amount'
]

# Load pre-trained model
try:
    iso_forest = joblib.load("D:\\DataScience\\Internshala\\New folder\\iso_forest_model1.pkl")
except FileNotFoundError as e:
    st.write(f"Error: {e}")
    st.write("Ensure the model file is in the correct directory.")
    st.stop()

# Streamlit app
st.title("Credit Card Fraud Detection")
st.write("Upload an Excel file with credit card transactions to detect frauds")

uploaded_file = st.file_uploader("Choose an Excel File", type=['xlsx', 'xls'])
if uploaded_file is not None:
    try:
        # Read the Excel file
        new_transactions = pd.read_excel(uploaded_file, sheet_name='creditcard_test')

        # Check for missing or extra columns
        missing_features = [feature for feature in feature_names if feature not in new_transactions.columns]
        extra_features = [col for col in new_transactions.columns if col not in feature_names]

        if missing_features:
            st.write(f"The uploaded file is missing the following required columns: {', '.join(missing_features)}")
            st.stop()
        elif extra_features:
            st.write(f"The uploaded file contains extra columns that are not expected: {', '.join(extra_features)}")
            st.stop()

        # Align the dataset with the expected features
        new_transactions = new_transactions[[feature for feature in feature_names if feature in new_transactions.columns]]

        # Clean the data: convert columns to numeric, coerce errors
        new_transactions = new_transactions.apply(pd.to_numeric, errors='coerce')

        # Remove rows with NaN values
        new_transactions = new_transactions.dropna()

        # Predict fraudulent transactions
        predictions = iso_forest.predict(new_transactions)
        predictions = pd.Series(predictions).map({1: 0, -1: 1})  # 1 for normal, -1 for anomaly
        new_transactions['anomaly'] = predictions

        # Display results
        fraudulent_transactions = new_transactions[new_transactions['anomaly'] == 1]
        st.write(f"Number of fraudulent transactions detected: {len(fraudulent_transactions)}")
        st.dataframe(fraudulent_transactions)

        # Visualization
        if not fraudulent_transactions.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='Amount', y='Time', hue='anomaly', data=fraudulent_transactions)
            plt.title('Detected Fraudulent Transactions')
            st.pyplot(plt)
    except Exception as e:
        st.write(f"An error occurred while processing the Excel file: {e}")
else:
    st.write("Please upload a file to proceed")
