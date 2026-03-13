import os
import sys
sys.path.insert(0, r"D:\Sys_Compression_Automatique")

from flask import Flask, request, jsonify
from agents.agent_analyseur  import AgentAnalyseur
from agents.agent_classifier import AgentClassifier

app = Flask(__name__)

# Initialiser les agents une seule fois au démarrage
agent1 = AgentAnalyseur()
agent2 = AgentClassifier()

print("=" * 50)
print("  API Agents - Compression Intelligente")
print("  Agent 1 : AgentAnalyseur     ✅")
print("  Agent 2 : AgentClassifier    ✅")
print("=" * 50)


# ------------------------------------------------------------------
# ROUTE 1 : Analyser une image (Agent 1)
# Entrée  : { "chemin_image": "D:\\...\\image.jpg" }
# Sortie  : rapport JSON complet (metadonnees, couleurs, complexite...)
# ------------------------------------------------------------------
@app.route("/analyser", methods=["POST"])
def analyser():
    try:
        data         = request.json
        chemin_image = data.get("chemin_image", "")

        if not chemin_image:
            return jsonify({
                "statut" : "erreur",
                "message": "Champ 'chemin_image' manquant"
            }), 400

        if not os.path.exists(chemin_image):
            return jsonify({
                "statut" : "erreur",
                "message": f"Image non trouvee : {chemin_image}"
            }), 404

        rapport = agent1.analyser(chemin_image)
        return jsonify(rapport), 200

    except Exception as e:
        return jsonify({
            "statut" : "erreur",
            "message": str(e)
        }), 500


# ------------------------------------------------------------------
# ROUTE 2 : Classifier une image (Agent 2)
# Entrée  : { "rapport": {...rapport agent1...}, "categorie": "photos" }
# Sortie  : recommandation JSON (format, qualite, justification, multi_llm)
# ------------------------------------------------------------------
@app.route("/classifier", methods=["POST"])
def classifier():
    try:
        data      = request.json
        rapport   = data.get("rapport", {})
        categorie = data.get("categorie", "photos")

        if not rapport:
            return jsonify({
                "statut" : "erreur",
                "message": "Champ 'rapport' manquant"
            }), 400

        # Ajouter la catégorie dans le rapport
        rapport["categorie"] = categorie

        recommandation = agent2.classifier(rapport)
        return jsonify(recommandation), 200

    except Exception as e:
        return jsonify({
            "statut" : "erreur",
            "message": str(e)
        }), 500


# ------------------------------------------------------------------
# ROUTE 3 : Pipeline complet Agent1 + Agent2 en un seul appel
# Entrée  : { "chemin_image": "...", "categorie": "photos" }
# Sortie  : { "analyse": {...}, "recommandation": {...} }
# ------------------------------------------------------------------
@app.route("/pipeline", methods=["POST"])
def pipeline():
    try:
        data         = request.json
        chemin_image = data.get("chemin_image", "")
        categorie    = data.get("categorie", "photos")

        if not chemin_image:
            return jsonify({
                "statut" : "erreur",
                "message": "Champ 'chemin_image' manquant"
            }), 400

        if not os.path.exists(chemin_image):
            return jsonify({
                "statut" : "erreur",
                "message": f"Image non trouvee : {chemin_image}"
            }), 404

        # Agent 1 : Analyse
        rapport              = agent1.analyser(chemin_image)
        rapport["categorie"] = categorie

        # Agent 2 : Classification
        recommandation = agent2.classifier(rapport)

        return jsonify({
            "statut"        : "succes",
            "image"         : chemin_image,
            "categorie"     : categorie,
            "analyse"       : rapport,
            "recommandation": recommandation
        }), 200

    except Exception as e:
        return jsonify({
            "statut" : "erreur",
            "message": str(e)
        }), 500


# ------------------------------------------------------------------
# ROUTE 4 : Vérifier que l API fonctionne
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "statut" : "ok",
        "message": "API Agents opérationnelle",
        "routes" : [
            "GET  /health      → vérifier que l API tourne",
            "POST /analyser    → Agent 1 : analyser une image",
            "POST /classifier  → Agent 2 : classifier + recommandation LLM",
            "POST /pipeline    → Agent 1 + Agent 2 en un seul appel"
        ]
    }), 200


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Démarrage de l API sur http://localhost:5000")
    print("   Routes disponibles :")
    print("   GET  http://localhost:5000/health")
    print("   POST http://localhost:5000/analyser")
    print("   POST http://localhost:5000/classifier")
    print("   POST http://localhost:5000/pipeline\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
