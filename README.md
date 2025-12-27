# SCFU — Streamlit Football Analytics App

## ▶️ Lancer l’application en local (Windows)

```powershell
# Activer l’environnement virtuel
.\venv\Scripts\Activate.ps1

# Lancer l’application Streamlit
streamlit run app_streamlit.py
📌 Présentation
SCFU est une application Streamlit d’analyse football basée sur des données événementielles et de tracking.
Elle permet de calculer et visualiser des indicateurs physiques et tactiques (IIC, phases de jeu, dynamiques temporelles, etc.).

L’application est conçue pour fonctionner :

soit avec des données locales (non versionnées)

soit via upload de fichiers ZIP (recommandé pour le déploiement)

🗂 Structure du projet
graphql
Copier le code
.
├── app_streamlit.py          # Application principale Streamlit
├── calc_iic.py               # Calculs des indicateurs (IIC, KPI)
├── pages/                    # Pages Streamlit additionnelles
├── requirements.txt          # Dépendances Python
├── README.md
├── .gitignore
└── data/                     # Données locales (ignorées par Git)
⚙️ Installation (première fois)
1️⃣ Créer un environnement virtuel
powershell
Copier le code
python -m venv venv
2️⃣ Activer l’environnement
powershell
Copier le code
.\venv\Scripts\Activate.ps1
3️⃣ Installer les dépendances
powershell
Copier le code
pip install -r requirements.txt
📥 Données
Les données ne sont pas versionnées dans le dépôt Git.

L’application attend généralement :

fichiers événementiels (*_dynamic_events.csv)

fichiers de tracking (*_tracking_extrapolated.jsonl)

métadonnées match (*_match.json, phases_of_play.csv)

Les données peuvent être :

stockées localement dans data/

ou fournies via upload ZIP depuis l’interface Streamlit

🚀 Déploiement (Streamlit Cloud)
Pour le déploiement :

seul le code est présent dans le repo

les données sont fournies à l’exécution

