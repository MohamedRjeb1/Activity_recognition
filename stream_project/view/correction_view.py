import streamlit as st
from controller import controller
import tempfile
def show_correction():
    st.header("🧘 Analyse et correction de posture")
    uploaded_file = st.file_uploader("Téléversez une vidéo pour correction", type=["mp4", "mov", "avi"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(video_path)
        if st.button("Corriger la posture"):
            feedback = controller.correct_posture(video_path)
            st.markdown("### Résultats de la correction :")
            st.write(feedback)
