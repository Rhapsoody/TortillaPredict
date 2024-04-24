# TortillaPredict
## API ENTRAÎNEMENT DE MODÈLE ET PRÉDICTION
> Bienvenue sur l'API d'entraînement de modèle et de prédiction des prix de tortillas au Mexique.  Cette API permet d'entraîner un modèle de prédiction des prix à partir de données fournies par l'utilisateur, ainsi que de prédire les prix futurs à partir de données d'entrée.

### Installation
Pour utiliser cette API, assurez-vous d'avoir Python installé sur votre système. Ensuite, suivez les étapes ci-dessous :

Cloner le dépôt GitHub : 
```git clone https://github.com/Rhapsoody/TortillaPredict.git```

Accédez au répertoire du projet : 
```cd TortillaPredict```

Installez les dépendances requises : 
```pip install -r requirements.txt```

Lancez l'API : 
```uvicorn api:app```

Lancez l'application : 
```streamlit run app.py```

### 1. Entraînement du Modèle

Endpoint : **/train**

Description : il permet d'entraîner un modèle de prédiction des prix à partir de données d'entraînement fournies par l'utilisateur.

Utilisez __tortilla_prices.csv__ ou un fichier ayant des données similaires pour entrainer le modèle. 
Les données seront nettoyés par algorithme pas besoin de le faire avant !

### 2. Prédiction des Prix
Endpoint : **/predict**

Description : il permet de prédire les prix futurs à partir de données d'entrée fournies par l'utilisateur.
Utilisez __tortilla_prices.csv__ ou un fichier ayant des données similaires pour prédire les prix. 
Les données seront nettoyés par algorithme pas besoin de le faire avant !

### 3. Question-Réponse avec OpenAI
Endpoint : **/model**

Description : il permet de poser des questions à un modèle OpenAI et d'obtenir des réponses.