import streamlit as st
from streamlit_webrtc import webrtc_streamer
import tempfile
from controller import controller
from model.real_time_detection import RealTimeProcessor

def show_detection():
    st.title("Interface Reconnaissance & Correction d'Activité")

    # Initialisation d'état si absent
    if "activity_summary" not in st.session_state:
        st.session_state.activity_summary = None
    if "start_camera" not in st.session_state:
        st.session_state.start_camera = False

    mode = st.radio("Choisissez le mode :", ["Vidéo Upload", "Caméra Temps Réel"])

    if mode == "Vidéo Upload":
        uploaded_file = st.file_uploader("Téléverser une vidéo", type=["mp4", "avi"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name

            st.video(video_path)

            if st.button("Analyser"):
                st.info("Analyse en cours...")
                activity_summary = controller.detect_activity(video_path)
                st.session_state.activity_summary = activity_summary

                if st.session_state.activity_summary :
                    st.subheader("Résumé des Activités Détectées")
                    for activity, count in st.session_state.activity_summary.items():
                        st.write(f"**{activity}** : {count} fois")
                else:
                    st.subheader("Pas d'activité détectée.")

        if st.session_state.activity_summary:
            if st.button("Enregistrer ce résumé"):
                if "userid" in st.session_state and st.session_state.userid:
                    controller.save_activity_summary(
                        st.session_state.userid,
                        st.session_state.activity_summary
                    )
                    st.success("Résumé enregistré avec succès !")
                    st.session_state.activity_summary = None
                else:
                    st.warning("Aucune donnée ou utilisateur non connecté.")
    
    elif mode == "Caméra Temps Réel":
        if st.button("Analyser"):
            st.session_state.start_camera = True

        if st.session_state.start_camera:
            webrtc_streamer(
                key="real-time-detection",
                video_processor_factory=RealTimeProcessor,
                media_stream_constraints={"video": True, "audio": False}
            )
