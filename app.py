import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model
model = joblib.load('xgboost_risk_model.pkl')

# App title and intro
st.set_page_config(page_title="Attawfiq Risk Scoring", layout="centered")
st.title("💳 Application de Scoring Crédit - Attawfiq Microfinance")
st.write("Ce prototype estime la probabilité qu’un nouveau client soit risqué en se basant sur le modèle XGBoost entraîné.")

# --- User Inputs ---
st.subheader("🧍‍♂️ Informations du Client")

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox("Genre", ["Homme", "Femme"])
    sit_fam = st.selectbox("Situation Familiale", ["Marie", "Celibataire", "Divorce", "Veuf"])
    niveau = st.selectbox("Niveau Scolaire", ["Analphabete", "Niveau Primaire", "Niveau Secondaire", "Niveau Superieur", "Non Renseigne"])
    activite = st.selectbox("Activité", ["Commerce", "Services", "Metiers Manuels", "Divers", "Autres"])
    logement = st.selectbox("Logement", ["Proprietaire", "A Construire", "Locataire", "Logement Parents", "Autre"])
with col2:
    zone = st.selectbox("Zone", ["Urbain", "Periurbain", "Rural"])
    age = st.number_input("Âge du client", min_value=18, max_value=80, value=35)
    nb_enf = st.number_input("Nombre d’enfants", min_value=0, max_value=10, value=2)
    mndeb = st.number_input("Montant débloqué (MAD)", min_value=500, max_value=200000, value=10000)
    duree = st.number_input("Durée du prêt (mois)", min_value=1, max_value=60, value=12)
periodicite = st.selectbox("Périodicité de remboursement", ["Mensuel", "Bimensuel", "Hebdomadaire"])

# --- Build Input DataFrame ---
client_data = pd.DataFrame({
    'Genre': [genre],
    'Situation_Familiale': [sit_fam],
    'Niveau_Scolaire': [niveau],
    'Activite': [activite],
    'Logement': [logement],
    'Zone': [zone],
    'AGE_CLT': [age],
    'NBRE_ENF': [nb_enf],
    'MNDEB': [mndeb],
    'duree_mois': [duree],
    'PERIODICITE': [periodicite]
})

st.write("### 🔍 Données saisies :")
st.dataframe(client_data)

# --- Encoding ---
client_encoded = pd.get_dummies(client_data)
# Align columns with the model
X_columns = model.get_booster().feature_names
for col in X_columns:
    if col not in client_encoded.columns:
        client_encoded[col] = 0
client_encoded = client_encoded[X_columns]

# --- Prediction ---
if st.button("Évaluer le Risque"):
    risk_proba = model.predict_proba(client_encoded)[:, 1][0]
    st.write(f"### Probabilité de Risque : **{risk_proba:.2%}**")

    if risk_proba < 0.3:
        st.success("🟢 Risque Faible – Client éligible au financement.")
    elif risk_proba < 0.6:
        st.warning("🟠 Risque Modéré – Vérification manuelle recommandée.")
    else:
        st.error("🔴 Risque Élevé – Prêt à accorder avec prudence.")
