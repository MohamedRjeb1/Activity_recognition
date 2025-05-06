import streamlit as st

def show_home():
    st.title("🏋️‍♂️ Sport Activity Recognition and Correction")
    st.markdown("Bienvenue dans l'application intelligente de reconnaissance et correction des mouvements sportifs ! 💪")
    st.subheader("📌 Objectif de l'application")
    st.markdown("""
    Cette application a pour but d'analyser en temps réel vos mouvements pendant l'entraînement
    et de vous fournir des retours sur la **précision et l'exécution de vos exercices** :
    
    - 🔍 Reconnaissance automatique des activités sportives
    - ✅ Analyse de la posture et détection des erreurs
    - 📈 Suggestions de correction pour une exécution optimale
    """)

    st.subheader("⚙️ Fonctionnalités disponibles")
    st.markdown("""
    - **Détection** : Analysez une activité à partir d'une vidéo ou caméra en direct.
    - **Correction** : Obtenez des retours personnalisés sur l'exécution de vos exercices.
    - **Historique (si implémenté)** : Consultez vos performances passées et vos progrès.
    """)

    st.subheader("🧭 Comment commencer ?")
    st.markdown("""
    1. Rendez-vous dans l'onglet **Détection** pour lancer une activité.
    2. Ensuite, passez à **Correction** pour voir les suggestions d'amélioration.
    3. Et surtout... entraînez-vous efficacement et en toute sécurité !
    """)

    st.success("👈 Utilisez la barre de navigation sur le côté pour commencer.")
