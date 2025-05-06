import streamlit as st
from view import layout
from controller import controller
import bcrypt #bibliothèque pour hacher les mots de passe de façon sécurisée.

# Initialisation des états de session
if "authenticated" not in st.session_state:#  session_state est un mécanisme de Streamlit pour stocker des données entre les rechargements de la page.
    st.session_state.authenticated = False #initailisation de variable  authenticated pour savoir si l’utilisateur est connecté.
    st.session_state.user_name = None#initialisation de nom de l'utisateur 
    st.session_state.userid=None
if "signup_mode" not in st.session_state:
    st.session_state.signup_mode = False #pour savoir si on est en mode inscription 

# Fonction d'inscription
def signup():
    # afficher un formulaire pour créer un compte.
    st.title("Créer un compte")
    name = st.text_input("Nom complet")
    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")
    confirm_password = st.text_input("Confirmer le mot de passe", type="password")
    if st.button("S'inscrire"):
        #verifier si tous les champs sont remplis
        if not (name and email and password and confirm_password):
            st.warning("Tous les champs sont requis.")
        elif password != confirm_password:
            st.warning("Les mots de passe ne correspondent pas.")
        elif controller.get_user_by_email(email):
            st.warning("Un utilisateur avec cet email existe déjà.")
        else:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            user_data = {
                "name": name,
                "email": email,
                "password": hashed_password.decode('utf-8')
            }
            controller.create_user(user_data)
            st.success("Compte créé avec succès ! Vous pouvez maintenant vous connecter.")
            st.session_state.signup_mode = False  # retour à la connexion

    if st.button("← Retour à la connexion"):
        st.session_state.signup_mode = False
        st.experimental_set_query_params()

# Fonction de connexion
def login():
    st.title("Connexion")
    #formulaire de connexion
    email = st.text_input("Email")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        user = controller.get_user_by_email(email)
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            st.session_state.authenticated = True
            st.session_state.user_name = user['name']
            st.session_state.userid = user['_id'] 
            st.success(f"Bienvenue {user['name']} !")
            st.rerun()
        else:
            st.error("Email ou mot de passe incorrect.")

    if st.button("Créer un compte"):
        st.session_state.signup_mode = True
        st.rerun()

# Fonction de déconnexion
def logout():
    st.session_state.authenticated = False
    st.session_state.user_name = None
    st.session_state.userid=None
    st.rerun()

# Interface principale 
if st.session_state.authenticated:
    st.sidebar.markdown(f"👤 {st.session_state.user_name}")
    if st.sidebar.button("Déconnexion"):
        logout()
    layout.run_app()
else:
    if st.session_state.signup_mode: #si l'utisateur a cliqué sur  "Créer un compte".
        signup()
    else:
        #a la premier ouverture de site la fonction appelé est login car authenticated et signup_mode sont  initialisés a false
        login()
