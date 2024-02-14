from flask import Flask, jsonify, render_template, request
import pandas as pd
import pickle
app = Flask(__name__)

# Chargement des données clients à partir d'un fichier CSV pour l'analyse et la prédiction
df = pd.read_csv('df_tabdashboard.csv')

# Chargement du modèle de machine learning pré-entraîné pour les prédictions de scoring de crédit
with open('model_streamlit.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Page d'accueil de l'application, affichant un formulaire pour entrer l'ID client
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Traitement de la requête de prédiction lorsque l'utilisateur soumet l'ID client
    if request.method == 'POST':
        # Extraction de l'ID client fourni via le formulaire
        client_id = int(request.form['client_id'])

        # Recherche de l'ID client dans le dataframe pour obtenir ses caractéristiques
        if client_id in df['SK_ID_CURR'].values:
            # Extraction des caractéristiques du client pour la prédiction
            client_features = df[df['SK_ID_CURR'] == client_id].values[0]

            # Utilisation du modèle pré-entraîné pour prédire l'état de crédit du client
            prediction = model.predict([client_features])[0]

            # Renvoi du résultat de la prédiction à l'utilisateur via une page HTML
            return render_template('result.html', prediction=prediction)
        else:
            # Informer l'utilisateur si l'ID client n'est pas trouvé dans les données
            return render_template('result.html', error="Identifiant client non trouvé dans nos enregistrements.")

    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # API pour effectuer une prédiction de scoring de crédit via une requête POST
    if request.method == 'POST':
        # Extraction de l'ID client à partir des données de la requête
        client_id = int(request.form['client_id'])

        # Vérification de l'existence de l'ID client dans le dataframe
        if client_id in df['SK_ID_CURR'].values:
            # Préparation des caractéristiques du client pour la prédiction
            client_features = df[df['SK_ID_CURR'] == client_id].values[0]

            # Prédiction en utilisant le modèle pré-entraîné
            prediction = model.predict([client_features])[0]

            # Retour du résultat de la prédiction en format JSON
            result = {'prediction': int(prediction)}
        else:
            # Retour d'une erreur si l'ID client n'est pas reconnu
            result = {'error': "ID client non reconnu dans nos données.", 'prediction': None}

        return jsonify(result)
    else:
        # Gestion des requêtes non-POST
        return jsonify({'error': "Méthode non supportée. Utilisez POST pour les prédictions."})
    
if __name__ == '__main__':
    app.run(port=8000)