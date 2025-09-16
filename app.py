import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load model and encoder
rf = joblib.load('rf_model.pkl')
enc = joblib.load('encoder.pkl')

st.title('CO2 Emissions Predictor for Green Supply Chains')

# File upload
uploaded_file = st.file_uploader('Upload Excel file with logistics data', type=['xlsx', 'xls'])
if uploaded_file:
    df_user = pd.read_excel(uploaded_file)
    
    # Assume columns: Distance, Weight, Vehicle_Type, Fuel_Efficiency
    if all(col in df_user.columns for col in ['Distance', 'Weight', 'Vehicle_Type', 'Fuel_Efficiency']):
        # Engineer features
        df_user['Consumption'] = df_user['Distance'] / df_user['Fuel_Efficiency']
        vt_enc_user = enc.transform(df_user[['Vehicle_Type']])
        vt_df_user = pd.DataFrame(vt_enc_user, columns=enc.get_feature_names_out(['Vehicle_Type']))
        df_eng_user = pd.concat([df_user[['Distance', 'Weight', 'Fuel_Efficiency', 'Consumption']], vt_df_user], axis=1)
        
        # Predict
        predictions = rf.predict(df_eng_user)
        df_user['Predicted_CO2'] = predictions
        
        st.success('Predictions complete!')
        st.dataframe(df_user.head())  # Show sample
        
        # Step 4: Beautiful Visualizations (Emission Hotspots)
        st.subheader('Emission Hotspots and Insights')
        
        # Bar chart: Total emissions by vehicle type (hotspots by category)
        grouped = df_user.groupby('Vehicle_Type')['Predicted_CO2'].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(grouped['Vehicle_Type'], grouped['Predicted_CO2'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Total Predicted CO2 Emissions by Vehicle Type (Hotspots)', fontsize=16)
        ax1.set_xlabel('Vehicle Type', fontsize=12)
        ax1.set_ylabel('Total CO2 (kg)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)
        
        # Scatter plot: Distance vs Predicted CO2, colored by vehicle (patterns/hotspots)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for vt in df_user['Vehicle_Type'].unique():
            subset = df_user[df_user['Vehicle_Type'] == vt]
            ax2.scatter(subset['Distance'], subset['Predicted_CO2'], label=vt, alpha=0.7)
        ax2.set_title('Predicted CO2 vs Distance (Colored by Vehicle Type)', fontsize=16)
        ax2.set_xlabel('Distance (km)', fontsize=12)
        ax2.set_ylabel('Predicted CO2 (kg)', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig2)
        
        # Optimization tip
        high_emissions = df_user[df_user['Predicted_CO2'] > df_user['Predicted_CO2'].quantile(0.75)]
        st.write(f'High-emission hotspots: {len(high_emissions)} entries above 75th percentile. Consider switching to lower-emission vehicles like Train/Ship.')
    else:
        st.error('Excel must have columns: Distance, Weight, Vehicle_Type, Fuel_Efficiency')