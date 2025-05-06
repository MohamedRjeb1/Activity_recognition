import streamlit as st
import cv2
import tempfile
from controller import controller
from datetime import datetime
# def show_detection_view():
#     st.title("Détection d'activité")

#     # Exemples d'activités et résultats
#     activity = st.selectbox("Choisir une activité", ["Squat", "Push-up", "shoulder_press",'barbell_bieceps_curl'])
#     prediction = st.text_input("Résultat de la détection", "Bonne exécution")

#     if st.button("Sauvegarder la détection"):
#         now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         controller.save_detection_result(activity, prediction, now)
#         st.success("Résultat enregistré dans la base de données.")
st.session_state.activity_summary=None
def show_detection():
 st.title("🧠 Interface Reconnaissance & Correction d'Activité")

 mode = st.radio("Choisissez le mode :", ["Vidéo Upload", "Caméra Temps Réel"])

 if mode == "Vidéo Upload":
    uploaded_file = st.file_uploader("📂 Téléverser une vidéo", type=["mp4", "avi"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        st.video(video_path)

        if st.button("Analyser"):
            st.info(" Analyse en cours...")
            activity_summary = controller.detect_activity(video_path)
            st.session_state.activity_summary = activity_summary
            if activity_summary is None:
             st.subheader("Pas d'activité détectée ")
            else :
             st.subheader("Résumé des Activités Détectées")
             for activity, count in activity_summary.items():
                 st.write(f"**{activity}** : {count} fois")
                 st.rerun()
    if "activity_summary" in st.session_state:
     if st.button("📌 Enregistrer ce résumé"):
                    if  st.session_state.userid and st.session_state.activity_summary :
                        controller.save_activity_summary(
                             st.session_state.userid,
                            st.session_state.activity_summary
                        )
                        st.success("Résumé enregistré avec succès !")
                        st.session_state.activity_summary=None
                    else:
                        st.warning("Vous n'avez pas des données a enregistrer.")
        

 else:
  if st.button("Analyser"):
    activity = controller.detect_activity(0)
    st.success(" Analyse terminée")
    st.subheader(" Activité détectée :")
    st.write(activity)
 return
