import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# Function to load models
def load_models():
    models = {}
    targets = ['AirTemp_Avg', 'BarPress_Avg', 'Rainfallrate_mm_Tot', 'RelativeHumidity',
               'SoilTemp_Avg', 'SolarRadiation_Avg', 'WindDir', 'WindSpeed_Avg', 'SoilMoisture']
    for target in targets:
        models[target] = {
            'rf': joblib.load(f'{target}_random_forest_model.pkl'),
            'lstm': load_model(f'{target}_lstm_model.h5'),
            'ann': load_model(f'{target}_ann_model.h5')
        }
    return models

# Ensemble prediction function
def predict_ensemble(models, features):
    results = {}
    for target, model_set in models.items():
        rf_pred = model_set['rf'].predict(features)
        lstm_pred = model_set['lstm'].predict(features).flatten()
        ann_pred = model_set['ann'].predict(features).flatten()
        ensemble_pred = np.mean([rf_pred, lstm_pred, ann_pred], axis=0)
        results[target] = ensemble_pred[0]
    return results

# Load the models
models = load_models()

# Streamlit interface
st.title('Weather Forecast Prediction')
st.markdown('### Enter the features to predict weather variables')

# Collecting user input features
Year = st.number_input('Year', min_value=2020, max_value=2030, value=2021)
Month = st.number_input('Month', min_value=1, max_value=12, value=1)
Day = st.number_input('Day', min_value=1, max_value=31, value=1)
DayOfWeek = st.number_input('Day of the Week', min_value=1, max_value=7, value=1)
WeekOfYear = st.number_input('Week of the Year', min_value=1, max_value=52, value=1)
Quarter = st.number_input('Quarter', min_value=1, max_value=4, value=1)

# Prepare feature array for prediction
features = np.array([[Year, Month, Day, DayOfWeek, WeekOfYear, Quarter]])

if st.button('Predict Weather'):
    results = predict_ensemble(models, features)
    st.write('### Prediction Results')
    st.json(results)

    # Download link for the results
    results_df = pd.DataFrame([results])
    st.download_button(
        label="Download prediction results",
        data=results_df.to_csv().encode('utf-8'),
        file_name='weather_predictions.csv',
        mime='text/csv',
    )

    # Plotting
    st.write('### Forecast Plot for the Year')
    fig, ax = plt.subplots()
    ax.plot(results.keys(), results.values(), marker='o')
    ax.set_title(f'Weather Forecast for {Year}')
    ax.set_xlabel('Weather Variables')
    ax.set_ylabel('Values')
    ax.grid(True)
    st.pyplot(fig)

    # Copyright notice
    st.text('Copyright Mathew Mark 2024')
