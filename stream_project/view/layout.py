import streamlit as st
from view.home import show_home
from view.detection_view import show_detection
from view.historique import show_historique
def run_app():
    if not st.session_state.get("authenticated"):
        st.warning("Vous devez être connecté pour accéder à l'application.")
        return
    menu = ["Accueil", "Lancer l'Entraînement", "historique_entrainement"]
    choice = st.sidebar.selectbox("Navigation", menu, key="sidebar_navigation")
    if choice == "Accueil":
        show_home()
    elif choice == "Lancer l'Entraînement":
        show_detection()
    elif choice == "historique_entrainement":
        show_historique()