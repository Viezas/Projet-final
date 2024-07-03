import streamlit as st
import requests
from pydantic import BaseModel

# Définir la classe de données pour Pydantic
class Data(BaseModel):
    age: int
    gender: int
    platform: int
    interests: int
    location: int
    demographics: int
    profession: int
    income: int
    indebt: int
    isHomeOwner: int
    Owns_Car: int

# Fonction pour appeler l'API de prédiction
def predict_screen_time(data: Data):
    url = "http://localhost:8000/predict"  # Modifier l'URL en fonction de votre environnement
    response = requests.post(url, json=data.dict())
    if response.status_code == 200:
        return response.json()["Prediction"]
    else:
        return None

# Interface utilisateur Streamlit
def main():
    st.title("Predict User Screen Time")

    # Section pour télécharger un fichier CSV pour entraîner le modèle
    st.header("Train Model with CSV")
    st.write("Upload a CSV file to train the model.")

    # Télécharger le fichier CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Bouton pour démarrer l'entraînement du modèle avec le fichier CSV
    if uploaded_file is not None:
        if st.button("Train Model"):
            # Appeler l'API de formation avec le fichier CSV
            url = "http://localhost:8000/training"
            files = {"file": uploaded_file}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                st.success("Model training completed successfully.")
            else:
                st.error("Failed to train the model. Please try again.")

    # Formulaire pour entrer les données utilisateur
    st.write("Enter user data to predict screen time:")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["male", "female", "non-binary"])
    platform = st.selectbox("Platform", ["Instagram", "Facebook", "YouTube"])
    interests = st.selectbox("Interests", ["Sports", "Travel", "Lifestyle"])
    location = st.text_input("Location", "United States")
    demographics = st.selectbox("Demographics", ["Urban", "Suburban", "Rural"])
    profession = st.selectbox("Profession", ["Software Engineer", "Student", "Marketing Manager"])
    income = st.number_input("Income", min_value=0, value=50000)
    indebt = 1 if st.checkbox("In Debt") else 0
    isHomeOwner = 1 if st.checkbox("Home Owner") else 0
    Owns_Car = 1 if st.checkbox("Owns Car") else 0

    if st.button("Predict"):
        # Mapper les valeurs de chaîne de caractères à des entiers
        gender_mapping = {"male": 0, "female": 1, "non-binary": 2}
        platform_mapping = {"Instagram": 0, "Facebook": 1, "YouTube": 2}  # Ajoutez d'autres plateformes au besoin
        interests_mapping = {"Sports": 0, "Travel": 1, "Lifestyle": 2}  # Ajoutez d'autres intérêts au besoin
        location_mapping = {"United Kingdom": 0, "Australia": 1, "United States": 2}  # Ajoutez d'autres pays au besoin
        demographics_mapping = {"Urban": 0, "Suburban": 1, "Rural": 2}  # Ajoutez d'autres types de zones au besoin
        profession_mapping = {"Software Engineer": 0, "Student": 1, "Marketing Manager": 2}  # Ajoutez d'autres professions au besoin

        # Utiliser le mapping pour convertir les chaînes de caractères en entiers
        gender = gender_mapping[gender]
        platform = platform_mapping[platform]
        interests = interests_mapping[interests]
        location = location_mapping[location]
        demographics = demographics_mapping[demographics]
        profession = profession_mapping[profession]

        # Créer l'objet Data
        data = Data(
            age=age,
            gender=gender,
            platform=platform,
            interests=interests,
            location=location,
            demographics=demographics,
            profession=profession,
            income=income,
            indebt=indebt,
            isHomeOwner=isHomeOwner,
            Owns_Car=Owns_Car
        )

        # Faire la prédiction
        prediction = predict_screen_time(data)
        if prediction is not None:
            st.success(f"Predicted screen time: {prediction}")
        else:
            st.error("Failed to make prediction. Please try again.")

    st.title("Téléchargement du modèle")

    # Bouton pour télécharger le modèle
    if st.button("Télécharger le modèle"):
        # Envoyer une requête GET à l'API FastAPI pour télécharger le modèle
        response = requests.get("http://localhost:8000/download-model")

        # Vérifier si la requête a réussi (statut 200)
        if response.status_code == 200:
            # Sauvegarder le contenu de la réponse dans un fichier model.h5
            with open("model.h5", "wb") as f:
                f.write(response.content)
            st.success("Modèle téléchargé avec succès sous le nom model.h5")
        else:
            st.error("Échec du téléchargement du modèle. Veuillez réessayer.")

if __name__ == "__main__":
    main()
