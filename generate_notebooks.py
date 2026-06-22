import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    }

# 1. EDA Notebook
eda_cells = [
    md_cell("# 01 - Analyse Exploratoire des Données (EDA) et Nettoyage\n\nCe notebook (brouillon) a pour but d'explorer le dataset des accidents de la route à Addis-Abeba, de comprendre les variables et de nettoyer les données."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Configuration de l'affichage\nsns.set_theme(style='whitegrid')\nimport warnings\nwarnings.filterwarnings('ignore')"),
    md_cell("## 1. Chargement des données"),
    code_cell("df = pd.read_csv('../../data/RTA Dataset.csv')\ndf.head()"),
    md_cell("## 2. Compréhension de base"),
    code_cell("df.info()"),
    code_cell("print(f'Nombre de lignes : {df.shape[0]}')\nprint(f'Nombre de colonnes : {df.shape[1]}')"),
    md_cell("Vérification de la variable cible : `Accident_severity`"),
    code_cell("df['Accident_severity'].value_counts()"),
    code_cell("sns.countplot(x='Accident_severity', data=df)\nplt.title('Distribution de la gravité des accidents')\nplt.show()"),
    md_cell("Comme on peut le voir, les classes sont très déséquilibrées (Slight injury est majoritaire)."),
    md_cell("## 3. Nettoyage et gestion des valeurs manquantes"),
    code_cell("missing_values = df.isnull().sum()\nmissing_values[missing_values > 0].sort_values(ascending=False)"),
    md_cell("On va remplir les valeurs manquantes avec le mode (valeur la plus fréquente) pour les variables catégorielles, ou simplement la catégorie 'Unknown'."),
    code_cell("def fill_missing(df):\n    df_clean = df.copy()\n    for col in df_clean.columns:\n        if df_clean[col].isnull().sum() > 0:\n            # Remplir par 'Unknown'\n            df_clean[col].fillna('Unknown', inplace=True)\n    return df_clean\n\ndf_clean = fill_missing(df)\nprint('Valeurs manquantes après nettoyage :', df_clean.isnull().sum().sum())"),
    md_cell("Sauvegarde des données nettoyées pour la modélisation."),
    code_cell("df_clean.to_csv('../../data/cleaned_data.csv', index=False)\nprint('Données nettoyées sauvegardées.')")
]
create_notebook("notebook/drafts/01_EDA_and_Cleaning.ipynb", eda_cells)

# 2. Modeling Notebook
mod_cells = [
    md_cell("# 02 - Modélisation ML Basique\n\nCe notebook explore des modèles simples pour prédire la gravité des accidents."),
    code_cell("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import classification_report, accuracy_score, f1_score\nimport warnings\nwarnings.filterwarnings('ignore')"),
    md_cell("## 1. Chargement des données nettoyées"),
    code_cell("df = pd.read_csv('../../data/cleaned_data.csv')\ndf.head()"),
    md_cell("## 2. Encodage des variables catégorielles\nLes modèles ML n'acceptent que des nombres. Nous allons utiliser `LabelEncoder` pour simplifier."),
    code_cell("le_dict = {}\ndf_encoded = df.copy()\n# On enlève la colonne Time car c'est une heure textuelle, on pourrait extraire l'heure mais on va la simplifier\ndf_encoded.drop('Time', axis=1, inplace=True)\n\nfor col in df_encoded.columns:\n    if df_encoded[col].dtype == 'object':\n        le = LabelEncoder()\n        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))\n        le_dict[col] = le"),
    md_cell("## 3. Séparation Train / Test"),
    code_cell("X = df_encoded.drop('Accident_severity', axis=1)\ny = df_encoded['Accident_severity']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint('Taille Train:', X_train.shape)\nprint('Taille Test:', X_test.shape)"),
    md_cell("## 4. Entraînement de modèles simples"),
    md_cell("### Régression Logistique"),
    code_cell("lr = LogisticRegression(max_iter=500)\nlr.fit(X_train, y_train)\ny_pred_lr = lr.predict(X_test)\nprint('Accuracy:', accuracy_score(y_test, y_pred_lr))\nprint(classification_report(y_test, y_pred_lr))"),
    md_cell("### Arbre de Décision (Decision Tree)"),
    code_cell("dt = DecisionTreeClassifier(random_state=42)\ndt.fit(X_train, y_train)\ny_pred_dt = dt.predict(X_test)\nprint('Accuracy:', accuracy_score(y_test, y_pred_dt))\nprint(classification_report(y_test, y_pred_dt))"),
    md_cell("L'arbre de décision capture mieux les relations non linéaires. C'est un modèle simple et interprétable, parfait pour répondre au cahier des charges.")
]
create_notebook("notebook/drafts/02_Modeling.ipynb", mod_cells)

# 3. Final Notebook
final_cells = [
    md_cell("# Projet Final Machine Learning : Prédiction de la Gravité des Accidents\n\n**Membres du groupe :** [Vos Noms]\n**Sujet :** Analyse des accidents de la route (Addis-Abeba)\n**Contexte :** Prédire si un accident est léger, grave ou fatal en fonction des conditions.\n\nCe notebook présente la démarche complète, de A à Z."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report, accuracy_score\nimport pickle\nimport warnings\nwarnings.filterwarnings('ignore')"),
    md_cell("## Étape 1 : Chargement et Nettoyage"),
    code_cell("df = pd.read_csv('../data/RTA Dataset.csv')\ndf.fillna('Unknown', inplace=True)\ndf.drop('Time', axis=1, inplace=True)\nprint('Données chargées et nettoyées. Taille :', df.shape)"),
    md_cell("## Étape 2 : Feature Engineering (Encodage)"),
    code_cell("le_dict = {}\ndf_encoded = df.copy()\nfor col in df_encoded.columns:\n    if df_encoded[col].dtype == 'object':\n        le = LabelEncoder()\n        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))\n        le_dict[col] = le"),
    md_cell("## Étape 3 : Modélisation (Random Forest Basique)"),
    code_cell("X = df_encoded.drop('Accident_severity', axis=1)\ny = df_encoded['Accident_severity']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Random Forest est simple mais puissant\nmodel = RandomForestClassifier(n_estimators=50, random_state=42)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(classification_report(y_test, y_pred))"),
    md_cell("## Étape 4 : Sauvegarde du modèle pour l'application Streamlit"),
    code_cell("with open('../streamlit_app/model.pkl', 'wb') as f:\n    pickle.dump(model, f)\n\nwith open('../streamlit_app/label_encoders.pkl', 'wb') as f:\n    pickle.dump(le_dict, f)\nprint('Modèle et encodeurs sauvegardés avec succès !')")
]
create_notebook("notebook/Final_Notebook.ipynb", final_cells)

print("Notebooks créés avec succès.")
