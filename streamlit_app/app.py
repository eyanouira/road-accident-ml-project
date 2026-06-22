import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Prédiction Gravité des Accidents", layout="centered")

st.title("🚗 Prédiction de la Gravité des Accidents de la Route")
st.write("Cette application prédit la gravité d'un accident en fonction des conditions de conduite.")

# Chargement du modèle et des encodeurs
@st.cache_resource
def load_components():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except FileNotFoundError:
        st.error("Les fichiers du modèle ne sont pas encore générés. Veuillez exécuter le Final_Notebook.ipynb d'abord.")
        return None, None

model, encoders = load_components()

if model and encoders:
    st.sidebar.header("Conditions de l'Accident")
    
    # Interface pour quelques variables importantes (simplifié pour l'exemple)
    day_of_week = st.sidebar.selectbox("Jour de la semaine", encoders['Day_of_week'].classes_)
    age_band = st.sidebar.selectbox("Age du conducteur", encoders['Age_band_of_driver'].classes_)
    weather = st.sidebar.selectbox("Conditions météo", encoders['Weather_conditions'].classes_)
    light = st.sidebar.selectbox("Luminosité", encoders['Light_conditions'].classes_)
    
    if st.sidebar.button("Prédire"):
        # Création d'un dictionnaire avec des valeurs par défaut ('Unknown' ou mode) pour toutes les autres features
        # Attention: pour un vrai déploiement, il faudrait collecter toutes les entrées requises par le modèle.
        input_data = {}
        for col, encoder in encoders.items():
            if col == 'Day_of_week': input_data[col] = day_of_week
            elif col == 'Age_band_of_driver': input_data[col] = age_band
            elif col == 'Weather_conditions': input_data[col] = weather
            elif col == 'Light_conditions': input_data[col] = light
            else:
                input_data[col] = encoder.classes_[0] # Valeur par défaut
        
        # Encodage des entrées
        df_input = pd.DataFrame([input_data])
        for col in df_input.columns:
            df_input[col] = encoders[col].transform(df_input[col])
            
        # Prédiction
        prediction = model.predict(df_input)[0]
        st.success(f"### Gravité prédite : **{prediction}**")
        st.info("💡 Attention : Cette prédiction est basée sur un modèle simplifié de Machine Learning.")
