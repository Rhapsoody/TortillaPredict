import streamlit as st
import requests

# URL de l'API
API_URL = "http://127.0.0.1:8000"

# Titre de l'application
st.title("Prédiction des prix de tortillas au Mexique")

# Sous-titre pour l'entraînement du modèle
st.header("Entraînement du modèle")

# Formulaire pour l'entraînement du modèle
train_data = st.text_input("Données d'entraînement (JSON)")

if st.button("Entraîner le modèle"):
    # Envoi de la requête à l'API
    response = requests.post(f"{API_URL}/train/", json=train_data)
    if response.status_code == 200:
        st.success("Modèle entraîné avec succès !")
    else:
        st.error("Erreur lors de l'entraînement du modèle.")

# Sous-titre pour la prédiction
st.header("Prédiction de prix")

# Formulaire pour la prédiction
prediction_data = st.text_input("Données de prédiction (JSON)")

if st.button("Prédire le prix"):
    # Envoi de la requête à l'API
    response = requests.post(f"{API_URL}/predict/", json=prediction_data)
    if response.status_code == 200:
        prediction_result = response.json()
        st.success(f"Prédiction réussie ! Résultat : {prediction_result}")
    else:
        st.error("Erreur lors de la prédiction.")

# Bouton pour récupérer le modèle
if st.button("Récupérer le modèle"):
    # Envoi de la requête à l'API
    response = requests.get(f"{API_URL}/model/")
    if response.status_code == 200:
        st.success("Modèle récupéré avec succès !")
    else:
        st.error("Erreur lors de la récupération du modèle.")
