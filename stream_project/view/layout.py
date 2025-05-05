# view/layout.py
import streamlit as st
from view.home import show_home
from view.detection_view import show_detection
from view.correction_view import show_correction

def run_app():
    menu = ["Accueil", "Détection", "Correction"]
    choice = st.sidebar.selectbox("Navigation", menu, key="sidebar_navigation")
    if choice == "Accueil":
        show_home()
    elif choice == "Détection":
        show_detection()
    elif choice == "Correction":
        show_correction()
