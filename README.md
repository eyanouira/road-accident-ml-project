# Road Accident Severity Prediction — Prédiction de la Gravité des Accidents
**Python • Jupyter • scikit-learn • Random Forest • Streamlit**

## 🎯 Présentation du Projet
Ce projet a pour objectif de prédire la gravité des accidents de la route (Slight, Serious, Fatal) à Addis-Abeba. En identifiant à l'avance les facteurs qui aggravent un accident, les autorités peuvent mettre en place des stratégies préventives pour améliorer la sécurité routière.

**Problème métier :** Les accidents de la route ont un coût humain et matériel énorme. Anticiper leur gravité en fonction des conditions (météo, âge, luminosité) est un levier stratégique pour la prévention et l'allocation des secours.

## 📁 Structure du Projet
```text
road-accident-ml/
│
├── 📓 Final_Notebook.ipynb                    # Notebook final soigné (démarche complète)
│
├── 📂 drafts/                                 # Notebooks de brouillon (expérimentations)
│   ├── 01_EDA_and_Cleaning.ipynb              # Analyse Exploratoire et Nettoyage
│   └── 02_Modeling.ipynb                      # Tests des modèles simples
│
├── 📂 data/
│   └── RTA Dataset.csv                        # Dataset source (Road Traffic Accidents)
│
├── 📂 streamlit_app/                          # Application Web de Déploiement
│   ├── app.py                                 # Code source Streamlit
│   ├── model.pkl                              # Modèle Random Forest sauvegardé
│   └── label_encoders.pkl                     # Encodeurs sauvegardés
│
├── 📂 presentation/
│   └── slides.md                              # Plan de la présentation PowerPoint (8 slides)
│
├── 🐍 run_all_models.py                       # Script Python (Clean code de tout le projet)
└── 📄 README.md                               # Ce fichier
```

## 📊 Dataset
**Source :** Road Traffic Accidents (Addis Ababa) — Kaggle

| Caractéristique | Valeur |
| :--- | :--- |
| **Nombre d'accidents** | 12 316 |
| **Nombre de variables** | 32 |
| **Variable cible** | `Accident_severity` (Slight, Serious, Fatal) |
| **Déséquilibre des classes** | ~84% Léger, ~14% Grave, ~1% Fatal |

**Variables clés :**
*   **Démographiques :** `Age_band_of_driver`, `Sex_of_driver`, `Educational_level`
*   **Environnement :** `Weather_conditions`, `Light_conditions`, `Road_surface_conditions`
*   **Accident :** `Number_of_vehicles_involved`, `Number_of_casualties`, `Cause_of_accident`
*   **Cible :** `Accident_severity` — La gravité mesurée de l'accident.

## 🗂️ Description des Notebooks

### 📓 Final_Notebook.ipynb — Le Notebook Principal
Ce notebook est le document final et propre du projet. Il regroupe l'intégralité de la démarche, divisée en 4 grandes parties :

🔍 **Partie 1 : Analyse Exploratoire (EDA)**
*   **Inspection :** Types de données, statistiques descriptives, valeurs manquantes.
*   **Variable cible :** Distribution de la gravité, impact très fort du déséquilibre des classes.

🛠️ **Partie 2 : Prétraitement des Données**
*   **Nettoyage :** Imputation des valeurs manquantes par la catégorie 'Unknown'.
*   **Préparation ML :** Suppression de la variable temporelle complexe (`Time`), Encodage des catégories avec `LabelEncoder`, et séparation Train/Test (80/20).

🤖 **Partie 3 : Modélisation (Machine Learning)**
*   **Machine Learning :** Entraînement de la Régression Logistique (baseline), d'un Arbre de Décision (Decision Tree), et d'une Forêt Aléatoire (Random Forest).

💾 **Partie 4 : Exportation**
*   Sauvegarde du meilleur modèle et de ses encodeurs pour l'application Streamlit.

### 📂 Dossier drafts/ — Les Brouillons d'Expérimentation
Ce dossier contient l'historique de recherche et les essais préliminaires :
*   `01_EDA_and_Cleaning.ipynb` : Les premiers tests de visualisation et de nettoyage brutal.
*   `02_Modeling.ipynb` : Les tests progressifs d'algorithmes et de métriques avant la consolidation finale.

## 📈 Résultats Comparatifs (Exemple de Précision)

| Modèle | Précision Globale (Accuracy) | Complexité |
| :--- | :--- | :--- |
| **Régression Logistique** | ~ 0.84 | Faible (Baseline) |
| **Arbre de Décision** | ~ 0.79 | Moyenne (Interprétable) |
| **Random Forest ⭐** | ~ 0.85 | Moyenne-Haute (Robuste) |

## 🏆 Modèle Final : Random Forest
Le **Random Forest** a été sélectionné comme modèle final car il offre la meilleure précision, est résistant au surapprentissage, et convient parfaitement à nos exigences de simplicité pour le projet.

## 🚀 Installation et Utilisation

**Prérequis :**
*   Python 3.9+
*   Jupyter Notebook ou VS Code

**Installation :**
```bash
# Installer les dépendances nécessaires
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

**Lancement :**
```bash
# Pour lancer le test automatisé des modèles :
python run_all_models.py

# Pour lancer l'application Web Streamlit :
cd streamlit_app
streamlit run app.py
```

**Ordre de lecture recommandé :**
1️⃣ `Final_Notebook.ipynb` → Démarche complète et conclusion (à lire en priorité).
2️⃣ `streamlit_app/app.py` → Application déployée interactive.
3️⃣ `drafts/01_EDA_and_Cleaning.ipynb` → Détail des premières explorations.

## 🛠️ Stack Technologique

| Outil | Usage |
| :--- | :--- |
| **Python 3.x** | Langage principal |
| **Pandas / NumPy** | Manipulation et calculs numériques |
| **Matplotlib / Seaborn** | Visualisation des données |
| **Scikit-learn** | Machine Learning, preprocessing, évaluation |
| **Streamlit** | Création de l'application Web interactive |
| **Pickle** | Sauvegarde et chargement du modèle |

## 📄 Licence / Crédits
Projet réalisé dans le cadre d'une étude de cas pédagogique en Machine Learning pour la prédiction d'accidents de la route, sous la supervision de M. Abdallah Khemais.

**Membres de l'équipe :**
*   Eya Nouira
