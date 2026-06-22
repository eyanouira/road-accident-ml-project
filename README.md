# 🚗 Road Accident Severity Prediction - Machine Learning End-to-End

Ce projet a été réalisé dans le cadre du module Machine Learning. Il vise à analyser les accidents de la route à Addis-Abeba et à prédire leur gravité en utilisant des algorithmes de Machine Learning.

## 📋 Contexte et Objectifs

- **Problématique** : Prédire la gravité d'un accident (Slight injury, Serious injury, Fatal injury).
- **Source des données** : [Road Traffic Accidents Dataset (Kaggle)](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents)
- **Taille du dataset** : ~12 300 enregistrements, 32 caractéristiques (météo, type de route, âge du conducteur, etc.).
- **Modèles testés** : Régression Logistique, Arbre de Décision, Random Forest.

## 🏗️ Architecture du Projet

Le dépôt est structuré de la manière suivante pour respecter les exigences du cahier des charges :

```text
├── data/
│   ├── RTA Dataset.csv         # Dataset original (à télécharger depuis Kaggle)
│   └── cleaned_data.csv        # Dataset après nettoyage et imputation
├── notebook/
│   ├── drafts/
│   │   ├── 01_EDA_and_Cleaning.ipynb  # Analyse exploratoire et nettoyage
│   │   └── 02_Modeling.ipynb          # Tests de modèles de Machine Learning
│   └── Final_Notebook.ipynb           # Notebook final structuré et commenté
├── presentation/
│   └── slides.md                      # Plan de la présentation PPT (7-8 slides)
├── streamlit_app/
│   ├── app.py                         # Code source de l'application web
│   ├── model.pkl                      # Modèle Random Forest entraîné
│   └── label_encoders.pkl             # Encodeurs pour les entrées utilisateur
└── README.md                          # Ce fichier
```

## 🛠️ Démarche et Pipeline Machine Learning

1. **Analyse Exploratoire (EDA)** : Visualisation de la distribution très déséquilibrée de la cible (les accidents légers sont majoritaires).
2. **Nettoyage des Données** : Imputation des valeurs manquantes par la classe majoritaire ou 'Unknown'. Suppression de colonnes non pertinentes.
3. **Feature Engineering** : Encodage des variables catégorielles avec `LabelEncoder`.
4. **Modélisation** : Entraînement de plusieurs modèles avec évaluation basée principalement sur le F1-Score (en raison du déséquilibre des classes). Le Random Forest a été retenu pour l'application finale.

## 🚀 Lancement de l'Application (Bonus)

L'application interactive Streamlit permet de prédire la gravité d'un accident en fonction de paramètres sélectionnés par l'utilisateur.

### Prérequis

Assurez-vous d'avoir Python installé ainsi que les librairies suivantes :
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

### Étapes

1. **Générer le modèle** : Ouvrez et exécutez entièrement le fichier `notebook/Final_Notebook.ipynb`. Cela va nettoyer les données, entraîner le Random Forest et générer les fichiers `.pkl` dans le dossier `streamlit_app/`.
2. **Lancer Streamlit** :
```bash
cd streamlit_app
streamlit run app.py
```
3. Une page web s'ouvrira automatiquement (par défaut `http://localhost:8501`).

## 👨‍💻 Membres du Groupe
- Eya Nouira
