# Système Intelligent de Compression d'Images 🖼️🤖

Ce projet est un système automatique et intelligent dédié à l'optimisation et la compression d'images. Basé sur une approche **Multi-Agents**, il combine le traitement d'images classique (OpenCV, Scikit-Image) avec des LLMs (Groq, Cohere, Mistral) pour choisir dynamiquement **le meilleur format (WebP, AVIF, HEIF, etc.)** et **la meilleure qualité** selon le contenu de l'image (Photo, Document, Screenshot...).

## 🌟 Architecture Multi-Agents

L'innovation principale de ce projet repose sur une architecture où 5 agents spécialisés collaborent de manière séquentielle pour garantir un résultat optimal.

### 1. 🔍 Agent Analyseur (Traitement d'Image)
Il agit comme les "yeux" du système. Avant même de penser à la compression, cet agent explore l'image pour en comprendre la nature mathématique et visuelle :
- **Analyse de la complexité** : Calcule l'entropie de Shannon pour évaluer la quantité d'informations (une image avec beaucoup de détails aura une forte entropie).
- **Textures (GLCM)** : Analyse la Matrice de Co-occurrence (contraste, corrélation et homogénéité de l'image).
- **Détection de texte (OCR)** : Repère la présence de texte via Tesseract. C'est crucial car le texte flou est inacceptable dans un document compressé.

### 2. 🤖 Agent Classificateur (Le Cerveau LLM)
C'est le décideur du système. Il reçoit le rapport détaillé de l'Agent Analyseur et la catégorie de l'image.
- **Vote Majoritaire Multi-LLM** : Fait appel simultanément à **3 Modèles de Langage (Groq Llama 3, Cohere, Mistral)**.
- **Prise de décision** : Chaque IA recommande un format cible (ex: WebP pour un screenshot, AVIF pour une photo détaillée) et un niveau de qualité.
- **Consensus** : L'agent synthétise les 3 réponses pour dégager une décision majoritaire (ex: "2 LLMs sur 3 recommandent WebP à 80%").

### 3. ⚙️ Agent Compresseur (L'Exécutant)
Il applique physiquement la décision prise par l'Agent Classificateur.
- Supporte un large éventail de formats modernes et classiques : **JPEG, PNG, WebP, AVIF, HEIF**.
- Réduit la taille du fichier en mémoire physique tout en respectant scrupuleusement le taux de qualité ordonné par les LLMs.

### 4. 📊 Agent Évaluateur (Le Contrôle Qualité)
Il compare mathématiquement l'image originale et l'image compressée pour s'assurer qu'il n'y a pas de perte visuelle inacceptable.
- **PSNR (Peak Signal) & MSE** : Évalue le bruit et l'erreur quadratique introduits par la compression.
- **SSIM (Structural Similarity)** : Mesure la perception de la dégradation structurale (luminance, contraste).
- **Taux de compression** : Calcule le gain en pourcentage (%) et en Ko par rapport au fichier d'origine.

### 5. 📝 Agent Rapporteur (Le Synthétiseur)
L'étape finale de l'orchestration.
- Génère un fichier **JSON complet** traçant tout l'historique de la décision (Pourquoi tel format a été choisi ? Quelles étaient les recommandations de Groq vs Mistral ?).
- Renvoie toutes ces données de manière structurée au Frontend Streamlit pour l'affichage des graphiques et des métriques.

## 🛠️ Technologies Utilisées
- **Frontend** : Streamlit
- **Backend** : Flask / API REST
- **Traitement d'Image** : OpenCV, Scikit-Image, Pillow (avec plugins HEIF/AVIF)
- **Modèles de Langage (LLM)** : Groq, Cohere, Mistral
- **Visualisation** : Matplotlib

## 📦 Installation et Prérequis

1. **Cloner le répertoire :**
   ```bash
   git clone https://github.com/ton-nom-utilisateur/Sys_Compression_Automatique.git
   cd Sys_Compression_Automatique
   ```

2. **Installer les dépendances :**
   Assurez-vous d'avoir Python installé, puis exécutez :
   ```bash
   pip install -r requirements.txt
   ```

3. **Variables d'Environnement :**
   Copiez le fichier `.env.example` en `.env` (ou créez-le) et ajoutez-y vos clés API :
   ```env
   GROQ_API_KEY=votre_cle_groq
   COHERE_API_KEY=votre_cle_cohere
   MISTRAL_API_KEY=votre_cle_mistral
   ```

## 🚀 Démarrage de l'Application

Le système fonctionne avec deux composants distincts fonctionnant en parallèle (Backend et Frontend).

**Terminal 1 : Lancer le Backend (API Flask)**
Ce serveur fait tourner l'intelligence artificielle et l'orchestration des agents.
```bash
python api.py
```

**Terminal 2 : Lancer le Frontend (Streamlit)**
Lancez l'interface graphique interactive pour l'utilisateur.
```bash
streamlit run app.py
```

## 📁 Structure et Fichiers Principaux

La séparation des fichiers garantit une architecture propre et modulaire (séparation Frontend / Backend) :

- **`api.py` (Le Backend / Cerveau)** : C'est le serveur Flask. Il ne possède aucune interface graphique. Son seul but est de recevoir les requêtes web, de déclencher les 5 agents Python de manière séquentielle, et de renvoyer le rapport final en JSON. C'est l'API REST de l'application.
- **`app_v1.py` (Ancienne Interface)** : C'est une version précédente (ou une variante simplifiée) du frontend Streamlit, probablement conservée pour des raisons de tests, de développement ou pour garder la trace d'une ancienne interface de validation.
- **`agents/`** : Dossier contenant l'intelligence artificielle pure (modules Python de chaque agent).
- **`dataset/` & `results/`** : Les répertoires de stockage des images originales en entrée et compressées en sortie, classées par dossier selon le type d'image.

### 📁Voici l'organisation de notre projet :

Sys_Compression_Automatique/

├── agents/                  # Intelligence artificielle pure (Modules Python des 5 agents)
├── dataset/                 # Répertoire des images originales en entrée
├── results/                 # Répertoire des images compressées en sortie et rapports JSON
├── .ipynb_checkpoints/      # Fichiers de sauvegarde Jupyter (Environnement de dev)
├── anaconda_projects/       # Configurations liées à Anaconda (Environnement de dev)
├── .env                     # Variables d'environnement (Clés API - NON PUBLIÉ SUR GIT)
├── .gitignore               # Fichiers et dossiers ignorés par Git (ex: .env, ngrok.exe)
├── api.py                   # Le Backend / Cerveau : Serveur Flask REST orchestrant les agents
├── app_v1.py                # Le Frontend : Interface utilisateur développée avec Streamlit
├── logo_fst.jpg             # Logo de l'Université Hassan II - FST Mohammedia
├── ngrok.exe                # Outil de tunneling pour exposer l'API localement sur le web
├── README.md                # Documentation principale du projet
└── requirements.txt         # Liste des dépendances et bibliothèques Python nécessaires
## 🎓 Contexte
Projet réalisé dans le cadre de l'Université Hassan II - FST Mohammedia.
