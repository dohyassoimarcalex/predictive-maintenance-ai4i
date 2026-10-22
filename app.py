import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

# Configuration de la page
st.set_page_config(page_title="PrediQ-Ops | Predictive Maintenance", layout="wide")

st.title("⚙️ PrediQ-Ops — Industrial Predictive Maintenance")
st.markdown("Surveillance d'équipement en temps réel & calcul de probabilité de panne.")

# 1. Chargement du modèle


@st.cache_resource
def load_model():
  # Cherche d'abord dans le dossier courant, sinon dans le home directory
  local_path = Path("model.pkl")
  home_path = Path.home() / "model.pkl"

  if local_path.exists():
    return joblib.load(local_path)
  return joblib.load(home_path)


model = load_model()

# 2. Entrées capteurs (Barre latérale)
st.sidebar.header("Paramètres des capteurs")

product_type = st.sidebar.selectbox("Qualité produit (Type)", ["L", "M", "H"])
air_temp = st.sidebar.slider("Température de l'air [K]", 295.0, 305.0, 298.0, 0.1)
process_temp = st.sidebar.slider("Température process [K]", 305.0, 315.0, 308.0, 0.1)
rpm = st.sidebar.slider("Vitesse de rotation [rpm]", 1100, 2900, 1500, 10)
torque = st.sidebar.slider("Couple [Nm]", 3.0, 80.0, 40.0, 0.5)
tool_wear = st.sidebar.slider("Usure outil [min]", 0, 260, 50, 1)

# 3. Calcul des variables physiques (Feature Engineering)
power_w = torque * (rpm * (2 * np.pi / 60))
temp_diff = process_temp - air_temp
wear_torque = tool_wear * torque

type_m = 1 if product_type == "M" else 0
type_l = 1 if product_type == "L" else 0

# Construction du vecteur d'entrée
input_data = pd.DataFrame([{
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Power [W]": power_w,
    "Temp_diff [K]": temp_diff,
    "Wear_Torque_interaction": wear_torque,
    "Type_M": type_m,
    "Type_L": type_l
}])

# Alignement strict sur les colonnes du modèle
input_data = input_data[model.feature_names_in_]

# 4. Affichage des grandeurs dérivées
col1, col2, col3 = st.columns(3)
col1.metric("Puissance mécanique", f"{power_w:.1f} W")
col2.metric("Gradient thermique (ΔT)", f"{temp_diff:.1f} K")
col3.metric("Indice contrainte outil", f"{wear_torque:.0f}")

st.divider()

# 5. Prédiction & Diagnostic
failure_prob = model.predict_proba(input_data)[0, 1]

st.subheader("Diagnostic opérationnel")
if failure_prob > 0.5:
    st.error(f"⚠️ **Risque critique de panne détecté : {failure_prob * 100:.1f} %**")
    st.write("Action recommandée : arrêt préventif et inspection de l'organe moteur.")
elif failure_prob > 0.2:
    st.warning(f"⚡ **Vigilance accrue requise : {failure_prob * 100:.1f} %**")
    st.write("Action recommandée : surveillance rapprochée des paliers d'usure.")
else:
    st.success(f"✅ **Fonctionnement nominal : {failure_prob * 100:.1f} % de risque**")
    st.write("Aucune anomalie détectée sur le cycle actuel.")