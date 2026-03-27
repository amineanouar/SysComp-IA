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
   git clone https://github.com/amineanouar/Sys_Compression_Automatique.git
   cd Sys_Compression_Automatique
