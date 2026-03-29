# Système Intelligent de Compression d'Images 🖼️🤖

Ce projet est un système automatique et intelligent dédié à l'optimisation et la compression d'images. Basé sur une approche **Multi-Agents**, il combine le traitement d'images classique (OpenCV, Scikit-Image) avec des LLMs (Groq, Cohere, Mistral) pour choisir dynamiquement **le meilleur format (WebP, AVIF, HEIF, etc.)** et **la meilleure qualité** selon le contenu de l'image (Photo, Document, Screenshot...).

## ✨ Fonctionnalités Principales

- **Automatisation avec n8n :** Le projet est entièrement conçu pour être orchestré via n8n (le fichier `workflow_agents.json` est inclus). Chaque étape est une route API distincte pour une agilité maximale.
- **Rétroaction Intelligente (Backtracking) :** Si la compression dégrade trop l'image (Score SSIM trop bas), le système s'auto-corrige et refait un essai avec une qualité supérieure avant de valider le résultat final.
- **Vote Majoritaire LLM :** Consensus entre 3 intelligences artificielles pour décider du meilleur format cible.
- **Analyse d'Image Poussée :** Détection de texte (OCR), analyse mathématique de la texture (GLCM) et de l'entropie.

---

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

---

## 🛠️ Technologies Utilisées
- **Orchestration Workflow** : n8n
- **Frontend UI** : Streamlit
- **Backend Serveur** : Flask / API REST
- **Traitement d'Image** : OpenCV, Scikit-Image, Pillow (avec plugins HEIF/AVIF), Tesseract OCR
- **Modèles de Langage (LLM)** : Groq, Cohere, Mistral

## 🎯 Formats Supportés

| Format | Type | Usage recommandé |
|---|---|---|
| JPEG | Avec pertes | Photos, paysages |
| PNG | Sans pertes | Documents, graphiques |
| WebP | Hybride | Screenshots, web |
| HEIF* | Avancé | Photos mobiles |
| AVIF* | Avancé | Compression maximale |

---

## 📦 Installation et Prérequis

**1. Cloner le répertoire :**
```bash
git clone https://github.com/amineanouar/Sys_Compression_Automatique.git
cd Sys_Compression_Automatique
```

**2. Installer Tesseract-OCR (Très important !) :**
L'Agent 1 utilise la détection de texte. Vous devez absolument avoir [Tesseract OCR installé sur votre machine Windows](https://github.com/UB-Mannheim/tesseract/wiki). Vérifiez ensuite que votre variable d'environnement `TESSERACT_CMD` pointe bien vers `.exe` dans votre fichier `.env`.

**3. Installer les dépendances Python :**
Assurez-vous d'avoir Python 3.9+ installé, puis exécutez :
```bash
pip install -r requirements.txt
```

**4. Variables d'Environnement :**
Copiez le modèle `.env.example` en le renommant `.env` et ajoutez-y vos clés API exactes :
```env
GROQ_API_KEY=votre_cle_groq
COHERE_API_KEY=votre_cle_cohere
MISTRAL_API_KEY=votre_cle_mistral

# Chemin d'installation de Tesseract sur votre machine (Modifiez si différent)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 🚀 Démarrage de l'Application

Le système fonctionne avec des composants distincts (API Backend, Frontend UI, et Automatisateur).

**Terminal 1 : Lancer le Backend (Le Cerveau - API Flask)**
Ce serveur expose les 5 agents via des routes REST (sans interface).
```bash
python api.py
```

**Terminal 2 : Lancer le Frontend V1 (Interface Utilisateur)**
Lancez l'interface graphique interactive pour envoyer manuellement une image.
```bash
streamlit run app_v1.py
```

*(Optionnel) Lancer le Workflow Automatisé n8n :* 
Importez le fichier `workflow/workflow_agents.json` dans votre logiciel n8n local pour utiliser l'automatisation de bout en bout de ces APIs Flask.

---

## 📁 Structure et Fichiers Principaux

La séparation des fichiers garantit une architecture propre et modulaire (API / UI / Agents) :

```text
Sys_Compression_Automatique/
├── agents/                  # IA pure (Modules Python des 5 agents indépendants)
│   ├── agent_analyseur.py   # Extraction (Entropie, GLCM, OCR Tesseract)
│   ├── agent_classifier.py  # Vote majoritaire LLM (Groq, Cohere, Mistral)
│   ├── agent_compresseur.py # Exécution de la compression
│   ├── agent_evaluateur.py  # Calcul (PSNR, SSIM) & Backtracking QA
│   └── agent_rapporteur.py  # Synthèse et JSON Final
├── dataset/                 # Base d'images de test pour le benchmark
├── results/                 # Sorties (Images finales compressées et JSON)
├── workflow/                # Scripts d'automatisation
│   └── workflow_agents.json # Template n8n pour orchestrer l'API Flask
├── .env.example             # Modèle vide pour les clés API sécurisées
├── api.py                   # Serveur REST Backend (L'intelligence)
├── app_v1.py                # Dashboard Frontend interactif sous Streamlit
├── ngrok.exe                # Tunneling (Si l'API est exposée en ligne)
├── README.md                # Documentation détaillée du projet
└── requirements.txt         # Librairies Python requises
```

## 🎓 Contexte
Projet académique réalisé dans le cadre de l'**Université Hassan II - FST Mohammedia**.
