import streamlit as st

def show_home():

    st.title("Sport Activity Recognition and Correction")
    st.markdown("Bienvenue dans l'application intelligente de reconnaissance et correction des mouvements sportifs ! ")
    st.subheader(" Objectif de l'application")
    st.markdown("""
    Cette application a pour but d'analyser en temps réel vos mouvements pendant l'entraînement
    et de vous fournir des retours sur la **précision et l'exécution de vos exercices** :
    
    - 🔍 Reconnaissance automatique des activités sportives
    - ✅ Analyse de la posture et détection des erreurs
    - 📈 Suggestions de correction pour une exécution optimale
    """)
