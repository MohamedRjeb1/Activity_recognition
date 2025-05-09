import streamlit as st
from controller import controller
import tempfile
def show_historique():

    st.header("Consulter votre historique")
    if st.button("Afficher mes résumés"):
     if st.session_state.userid:
        summaries = controller.get_activity_summary(st.session_state.userid)
        if summaries:
            total_counts = {}

            for summary in summaries:
                for activity, count in summary['summary'].items():
                    if activity in total_counts:
                        total_counts[activity] += count
                    else:
                        total_counts[activity] = count

            st.subheader("Somme totale par activité :")
            for activity, total in total_counts.items():
                st.write(f"**{activity}** : {total} fois")
        else:
            st.write("Erreur")
     else:
        st.warning("Vous êtes non connecté.")


