import streamlit as st
import requests
import json
import os
import tempfile
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
try:
    import pillow_avif
except ImportError:
    pass
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Compression Intelligente d'Images",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL     = "http://localhost:5000"
WEBHOOK_N8N = "https://n8n.ismailslab.com/webhook/compression"
TIMEOUT     = 300
MAX_PIXELS  = 4000 * 4000
MODE        = "flask"

# ============================================================
# INITIALISATION SESSION STATE
# ============================================================
if "resultats_prets" not in st.session_state:
    st.session_state["resultats_prets"]     = False
    st.session_state["rapport_final"]       = None
    st.session_state["rapport_evaluation"]  = None
    st.session_state["recommandation"]      = None
    st.session_state["rapport_analyse"]     = None
    st.session_state["chemin_meilleure"]    = ""
    st.session_state["taille_originale_kb"] = 0
    st.session_state["nom_image"]           = ""

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BANNIERE
# ============================================================
chemin_banniere = r"D:\Sys_Compression_Automatique\banner.png"
if os.path.exists(chemin_banniere):
    st.image(chemin_banniere, use_container_width=True)
else:
    st.markdown('<div class="main-title">🖼️ Système Intelligent de Compression d\'Images</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Approche Multi-Agents et IA Générative — FST Mohammedia — Licence IRM 2025-2026</div>',
                unsafe_allow_html=True)
st.divider()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    logo_fst = r"D:\Sys_Compression_Automatique\assets\logo_fst.jpg"
    if os.path.exists(logo_fst):
        st.image(logo_fst, width=160)
    else:
        st.markdown("### 🎓 FST Mohammedia")
    st.caption("Université Hassan II de Casablanca")

    st.divider()
    st.header("⚙️ Configuration")
    categorie = st.selectbox(
        "📁 Catégorie de l'image",
        ["photos", "documents", "graphiques", "screenshots"]
    )

    st.divider()
    st.header("🔧 Mode de traitement")
    mode_choisi = st.radio(
        "Choisir le pipeline :",
        ["Flask direct (local)", "n8n (workflow cloud)"],
        index=0
    )
    if "n8n" in mode_choisi:
        MODE = "n8n"
        st.info("📡 Les agents seront appelés via n8n")
    else:
        MODE = "flask"
        st.info("⚡ Les agents seront appelés directement via Flask")

    st.divider()
    st.header("🤖 LLMs Utilisés")
    st.markdown("**Groq** / Llama3-70b")
    st.markdown("**Cohere** / Command-R")
    st.markdown("**Mistral** / mistral-small")
    st.caption("Vote majoritaire entre les 3 LLMs")

    st.divider()
    st.header("📊 Métriques")
    st.markdown("- **PSNR** : Qualité signal (dB)")
    st.markdown("- **SSIM** : Similarité structurelle")
    st.markdown("- **MSE** : Erreur quadratique")
    st.markdown("- **Taux** : Réduction de taille (%)")

    st.divider()
    try:
        r_health = requests.get(f"{API_URL}/health", timeout=3)
        if r_health.status_code == 200:
            st.success("✅ API Flask connectée")
        else:
            st.error("❌ API Flask non disponible")
    except:
        st.error("❌ API Flask non disponible\nLancez api.py d'abord !")

# ============================================================
# ZONE UPLOAD
# ============================================================
col_upload, col_config = st.columns([2, 1])

with col_upload:
    st.subheader("📤 Uploader une image")
    fichier_upload = st.file_uploader(
        "Glissez-déposez ou cliquez pour choisir",
        type=["jpg", "jpeg", "png", "tif", "tiff", "webp", "bmp"]
    )

with col_config:
    st.subheader("🎯 Options")
    afficher_details = st.checkbox("Afficher les features extraites", value=False)
    afficher_json    = st.checkbox("Afficher le rapport JSON",        value=False)
    comparer_formats = st.checkbox("Comparer tous les formats",       value=True)

    # ← NOUVEAU
    st.divider()
    st.markdown("**📐 Redimensionnement**")
    forcer_dimension = st.checkbox("Forcer une dimension fixe", value=False)
    if forcer_dimension:
        dim_cible = st.selectbox(
            "Dimension cible",
            ["1920x1080 (Full HD)", "1280x720 (HD)", "800x600", "512x512 (Carré)"]
        )
        dim_map = {
            "1920x1080 (Full HD)": (1920, 1080),
            "1280x720 (HD)"      : (1280, 720),
            "800x600"            : (800, 600),
            "512x512 (Carré)"    : (512, 512)
        }
        target_dim = dim_map[dim_cible]
    else:
        target_dim = None

# ============================================================
# AFFICHAGE IMAGE UPLOADEE
# ============================================================
if fichier_upload is not None:

    if fichier_upload.name != st.session_state["nom_image"]:
        st.session_state["resultats_prets"]     = False
        st.session_state["rapport_final"]       = None
        st.session_state["rapport_evaluation"]  = None
        st.session_state["recommandation"]      = None
        st.session_state["rapport_analyse"]     = None
        st.session_state["chemin_meilleure"]    = ""
        st.session_state["nom_image"]           = fichier_upload.name

    image_pil           = Image.open(fichier_upload)
    taille_originale_kb = len(fichier_upload.getvalue()) / 1024

    if image_pil.width * image_pil.height > MAX_PIXELS:
        st.warning(f"⚠️ Image très grande ({image_pil.width}x{image_pil.height} px) "
                   f"— redimensionnée automatiquement avant compression.")

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_pil, caption=f"Image originale — {fichier_upload.name}", width=400)
    with col2:
        st.markdown("### 📋 Informations")
        st.markdown(f"**📐 Résolution :** {image_pil.width} x {image_pil.height} px")
        st.markdown(f"**📦 Taille :** {taille_originale_kb:.1f} KB")
        st.markdown(f"**🎨 Mode couleur :** {image_pil.mode}")
        st.markdown(f"**📁 Format :** {fichier_upload.name.split('.')[-1].upper()}")

    st.divider()

    if st.button("🚀 Lancer la compression intelligente", type="primary", use_container_width=True):

        chemin_tmp          = None
        rapport_analyse     = {}
        recommandation      = {}
        rapport_compression = {}
        rapport_evaluation  = {}
        rapport_final       = {}

        try:
            os.makedirs(r"D:\Sys_Compression_Automatique\dataset\temp", exist_ok=True)
            os.makedirs(r"D:\Sys_Compression_Automatique\results\temp",  exist_ok=True)

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{fichier_upload.name.split('.')[-1]}",
                dir=r"D:\Sys_Compression_Automatique\dataset\temp"
            ) as tmp:
                tmp.write(fichier_upload.getvalue())
                chemin_tmp = tmp.name

            img_check = Image.open(chemin_tmp)
            from PIL import ImageOps
            img_check = Image.open(chemin_tmp).convert("RGB")

            # Redimensionnement 4K si trop grande
            if img_check.width * img_check.height > MAX_PIXELS:
                ratio     = (MAX_PIXELS / (img_check.width * img_check.height)) ** 0.5
                new_w     = int(img_check.width  * ratio)
                new_h     = int(img_check.height * ratio)
                img_check = img_check.resize((new_w, new_h), Image.LANCZOS)
                st.info(f"📐 Redimensionnée 4K : {new_w}x{new_h} px")

            # Redimensionnement à dimension cible si demandé
            if target_dim is not None and img_check.size != target_dim:
                img_check = ImageOps.fit(
                    img_check,
                    target_dim,
                    method    = Image.LANCZOS,
                    centering = (0.5, 0.5)
                )
                st.info(f"📐 Redimensionnée → {target_dim[0]}x{target_dim[1]} px")

            img_check.save(chemin_tmp, "JPEG", quality=95)
            img_check.close()

            # ============================================================
            # MODE N8N
            # ============================================================
            if MODE == "n8n":
                with st.spinner("🔄 Pipeline n8n en cours — 5 agents en exécution..."):
                    r_n8n = requests.post(
                        WEBHOOK_N8N,
                        json={"chemin_image": chemin_tmp, "categorie": categorie},
                        timeout=TIMEOUT
                    )

                if not r_n8n.text or r_n8n.text.strip() == "":
                    st.error("❌ n8n retourne une réponse vide — vérifie Respond to Webhook")
                    st.stop()

                if r_n8n.status_code != 200:
                    st.error(f"❌ Erreur n8n : status {r_n8n.status_code}")
                    st.stop()

                # ── Fonction utilitaire : convertir string → dict ────────
                def parse(val):
                    """Convertit une string JSON en dict si nécessaire."""
                    if val is None:
                        return {}
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except:
                            return {}
                    if isinstance(val, dict):
                        return val
                    if isinstance(val, list) and len(val) > 0:
                        return val[0] if isinstance(val[0], dict) else {}
                    return {}

                # ── n8n retourne une liste → prendre le 1er élément ─────
                raw = r_n8n.json()
                if isinstance(raw, list):
                    raw = raw[0] if raw else {}

                # ── Extraire chaque rapport ──────────────────────────────
                rapport_analyse     = parse(raw.get("rapport_analyse"))
                recommandation      = parse(raw.get("recommandation"))
                rapport_compression = parse(raw.get("rapport_compression"))
                rapport_evaluation  = parse(raw.get("rapport_evaluation"))
                rapport_final       = parse(raw.get("rapport_final"))

                # ── Si les clés n'existent pas → n8n retourne Agent5 direct
                if not rapport_evaluation:
                    # n8n retourne directement le rapport Agent5
                    rapport_final      = raw
                    detail_evals       = raw.get("detail_evaluations", [])
                    meilleure_eval     = {}
                    if detail_evals:
                        # Chercher celle avec label "recommande"
                        for e in detail_evals:
                            if e.get("label") == "recommande":
                                meilleure_eval = e
                                break
                        if not meilleure_eval:
                            meilleure_eval = detail_evals[0]

                    rapport_evaluation = {
                        "meilleure_compression": meilleure_eval,
                        "evaluations"          : detail_evals
                    }

                    rec_llm = raw.get("recommandation_llm", {})
                    recommandation = {
                        "format_recommande"   : rec_llm.get("format_recommande"),
                        "qualite_recommandee" : rec_llm.get("qualite_recommandee"),
                        "justification"       : rec_llm.get("justification"),
                        "multi_llm"           : rec_llm.get("multi_llm", {})
                    }

                    img_info = raw.get("image", {})
                    rapport_analyse = {
                        "complexite": {"niveau_complexite": img_info.get("complexite", ""),
                                       "score_complexite" : img_info.get("score_complexite", 0)},
                        "textures"  : {},
                        "ocr"       : {}
                    }

                st.success("✅ Pipeline n8n terminé avec succès !")
                st.balloons()

            # ============================================================
            # MODE FLASK DIRECT
            # ============================================================
            else:
                with st.spinner("🔍 Agent 1 — Analyse des caractéristiques..."):
                    r1 = requests.post(f"{API_URL}/analyser",
                                       json={"chemin_image": chemin_tmp}, timeout=TIMEOUT)
                    rapport_analyse              = r1.json()
                    rapport_analyse["categorie"] = categorie
                st.success("✅ Agent 1 — Analyse terminée")

                with st.spinner("🤖 Agent 2 — Consultation des 3 LLMs..."):
                    r2 = requests.post(f"{API_URL}/classifier",
                                       json={"rapport": rapport_analyse, "categorie": categorie},
                                       timeout=TIMEOUT)
                    recommandation = r2.json()
                st.success("✅ Agent 2 — Recommandation obtenue")

                with st.spinner("⚙️ Agent 3 — Compression en cours..."):
                    r3 = requests.post(f"{API_URL}/compresser", json={
                        "chemin_image"       : chemin_tmp,
                        "format_recommande"  : recommandation.get("format_recommande",   "JPEG"),
                        "qualite_recommandee": recommandation.get("qualite_recommandee", 85),
                        "dossier_sortie"     : r"D:\Sys_Compression_Automatique\results\temp"
                    }, timeout=TIMEOUT)
                    rapport_compression = r3.json()
                st.success("✅ Agent 3 — Compression terminée")

                with st.spinner("📊 Agent 4 — Calcul des métriques..."):
                    r4 = requests.post(f"{API_URL}/evaluer", json={
                        "chemin_originale"   : chemin_tmp,
                        "rapport_compression": rapport_compression
                    }, timeout=TIMEOUT)
                    rapport_evaluation = r4.json()
                st.success("✅ Agent 4 — Évaluation terminée")

                with st.spinner("📝 Agent 5 — Génération du rapport..."):
                    r5 = requests.post(f"{API_URL}/rapport", json={
                        "rapport_analyse"    : rapport_analyse,
                        "recommandation"     : recommandation,
                        "rapport_compression": rapport_compression,
                        "rapport_evaluation" : rapport_evaluation
                    }, timeout=TIMEOUT)
                    rapport_final = r5.json()
                st.success("✅ Agent 5 — Rapport généré")
                st.balloons()

            # ── Sauvegarder dans session_state ───────────────────────────
            st.session_state["resultats_prets"]     = True
            st.session_state["rapport_final"]       = rapport_final
            st.session_state["rapport_evaluation"]  = rapport_evaluation
            st.session_state["recommandation"]      = recommandation
            st.session_state["rapport_analyse"]     = rapport_analyse
            st.session_state["taille_originale_kb"] = taille_originale_kb
            
            chem_meill = rapport_evaluation.get(
                "meilleure_compression", {}
            ).get("chemin_fichier", "") if isinstance(rapport_evaluation, dict) else ""
            
            if chem_meill and os.path.exists(chem_meill):
                import shutil
                nom_base = fichier_upload.name.rsplit('.', 1)[0]
                ext_comp = chem_meill.split('.')[-1]
                nom_fichier = f"{nom_base}_comp.{ext_comp}"
                
                dossier_res = r"D:\Sys_Compression_Automatique\results"
                dossier_cat = os.path.join(dossier_res, categorie)
                os.makedirs(dossier_cat, exist_ok=True)
                
                chemin_c = os.path.join(dossier_cat, nom_fichier)
                
                shutil.copy2(chem_meill, chemin_c)
                st.session_state["chem_final"] = chemin_c
                chem_meill = chemin_c
            st.session_state["chemin_meilleure"]    = chem_meill

        except requests.exceptions.ConnectionError:
            st.error("❌ API Flask non disponible — lancez api.py d'abord !")
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout — essayez avec une image plus petite.")
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {str(e)}")
        finally:
            if chemin_tmp and os.path.exists(chemin_tmp):
                try:
                    os.remove(chemin_tmp)
                except:
                    pass

# ============================================================
# AFFICHAGE DES RESULTATS
# ============================================================
if st.session_state["resultats_prets"]:

    rapport_final       = st.session_state["rapport_final"]
    rapport_evaluation  = st.session_state["rapport_evaluation"]
    recommandation      = st.session_state["recommandation"]
    rapport_analyse     = st.session_state["rapport_analyse"]
    taille_originale_kb = st.session_state["taille_originale_kb"]
    chemin_meilleure    = st.session_state["chemin_meilleure"]
    meilleure = rapport_evaluation.get("meilleure_compression", {}) if isinstance(rapport_evaluation, dict) else {}

    st.divider()
    st.header("📊 Résultats de la Compression")

    # ── Recommandation Multi-LLM ─────────────────────────────
    st.subheader("🤖 Recommandation Multi-LLM")
    multi = recommandation.get("multi_llm", {}) if isinstance(recommandation, dict) else {}
    col_g, col_c, col_m, col_res = st.columns(4)

    with col_g:
        st.markdown("**🤖 Groq / Llama3**")
        fmt = multi.get('llm1_groq', {}).get('format', 'N/A')
        qlt = multi.get('llm1_groq', {}).get('qualite', 'N/A')
        st.info(f"Format recommandé : **{fmt}**\n\nQualité recommandée : **{qlt}%**")
    with col_c:
        st.markdown("**🤖 Cohere / Command-R**")
        fmt = multi.get('llm2_cohere', {}).get('format', 'N/A')
        qlt = multi.get('llm2_cohere', {}).get('qualite', 'N/A')
        st.info(f"Format recommandé : **{fmt}**\n\nQualité recommandée : **{qlt}%**")
    with col_m:
        st.markdown("**🤖 Mistral / mistral-small**")
        fmt = multi.get('llm3_mistral', {}).get('format', 'N/A')
        qlt = multi.get('llm3_mistral', {}).get('qualite', 'N/A')
        st.info(f"Format recommandé : **{fmt}**\n\nQualité recommandée : **{qlt}%**")
    with col_res:
        st.markdown("**Décision finale**")
        accord        = multi.get("accord", "")
        format_final  = recommandation.get("format_recommande",   "N/A") if isinstance(recommandation, dict) else "N/A"
        qualite_final = recommandation.get("qualite_recommandee", "N/A") if isinstance(recommandation, dict) else "N/A"
        if accord == "accord_total":
            st.success(f"✅ Format : {format_final}\n\nQualité : {qualite_final}%\n\n🤝 Les 3 LLMs sont unanimes !")
        elif "2_sur_3" in str(accord):
            st.warning(f"⚠️ Format : {format_final}\n\nQualité : {qualite_final}%\n\n🗳️ 2 LLMs sur 3 sont d'accord")
        else:
            st.error(f"❌ Format : {format_final}\n\nQualité : {qualite_final}%\n\n⚡ Désaccord total")

    if isinstance(recommandation, dict) and recommandation.get("justification"):
        st.info(f"💬 **Justification LLM :** {recommandation.get('justification', '')[:400]}")

    st.divider()

    # ── Métriques ────────────────────────────────────────────
    st.subheader("📈 Métriques de Qualité")
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("🏆 Meilleur format", meilleure.get("format", "N/A"))
    with col_m2:
        psnr = meilleure.get("psnr_db", 0)
        st.metric("📡 PSNR", f"{psnr} dB",
                  delta="Excellent" if psnr >= 40 else "Bon" if psnr >= 35 else "Acceptable")
    with col_m3:
        ssim = meilleure.get("ssim", 0)
        st.metric("🔍 SSIM", f"{ssim}",
                  delta="Excellent" if ssim >= 0.95 else "Bon" if ssim >= 0.90 else "Acceptable")
    with col_m4:
        st.metric("📉 Taux compression", f"{meilleure.get('taux_compression_pct', 0)}%")
    with col_m5:
        taille_finale = meilleure.get("taille_compresse_kb", 0)
        st.metric("💾 Taille finale", f"{taille_finale:.1f} KB",
                  delta=f"-{taille_originale_kb - taille_finale:.1f} KB")

    st.divider()

    # ── Comparaison Avant / Après ─────────────────────────────
    st.subheader("🖼️ Comparaison Avant / Après")
    if chemin_meilleure and os.path.exists(chemin_meilleure):
        col_avant, col_apres = st.columns(2)
        with col_avant:
            if fichier_upload is not None:
                st.image(Image.open(fichier_upload),
                         caption=f"Originale — {taille_originale_kb:.1f} KB", width=500)
        with col_apres:
            try:
                img_comp    = Image.open(chemin_meilleure).convert("RGB")
                taille_comp = os.path.getsize(chemin_meilleure) / 1024
                st.image(img_comp,
                         caption=f"Compressée ({meilleure.get('format')}) — {taille_comp:.1f} KB",
                         width=500)
            except:
                taille_comp = os.path.getsize(chemin_meilleure) / 1024
                st.warning(f"⚠️ Aperçu non disponible pour {meilleure.get('format')}")
                st.info(f"✅ Fichier créé : {taille_comp:.1f} KB — téléchargez-le ci-dessous.")

    st.divider()

    # ── Graphiques ────────────────────────────────────────────
    if comparer_formats and isinstance(rapport_evaluation, dict):
        st.subheader("📊 Comparaison de tous les formats")
        evaluations = rapport_evaluation.get("_toutes_evaluations", rapport_evaluation.get("evaluations", []))
        if evaluations:
            formats  = [f"{e['format']} q={e['qualite']}" for e in evaluations]
            psnr_v   = [e["psnr_db"]              for e in evaluations]
            ssim_v   = [e["ssim"]                 for e in evaluations]
            taux_v   = [e["taux_compression_pct"] for e in evaluations]
            scores_v = [e["score_global"]          for e in evaluations]
            x        = np.arange(len(formats))
            colors   = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"][:len(formats)]

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.patch.set_facecolor('#f8f9fa')
            for ax, vals, title, ylabel, seuil in [
                (axes[0], psnr_v,   "PSNR (dB)",          "dB",    35),
                (axes[1], ssim_v,   "SSIM",               "Score", 0.90),
                (axes[2], taux_v,   "Taux compression %", "%",     None),
                (axes[3], scores_v, "Score Global /100",  "Score", None)
            ]:
                bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=1.5)
                if seuil:
                    ax.axhline(y=seuil, color="red", linestyle="--", alpha=0.7,
                               label=f"Seuil {seuil}")
                    ax.legend(fontsize=8)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(formats, rotation=15, fontsize=8)
                ax.set_ylabel(ylabel)
                ax.set_facecolor('#f8f9fa')
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{val:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
            plt.suptitle("Comparaison des compressions", fontsize=13, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

    st.divider()

    # ── Téléchargement ────────────────────────────────────────
    st.subheader("📥 Télécharger les fichiers")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if chemin_meilleure and os.path.exists(chemin_meilleure):
            with open(chemin_meilleure, "rb") as f:
                ext  = chemin_meilleure.split(".")[-1].lower()
                mime = "image/jpeg" if ext == "jpg" else f"image/{ext}"
                st.download_button(
                    label=f"⬇️ Image compressée ({meilleure.get('format')})",
                    data=f.read(),
                    file_name=os.path.basename(chemin_meilleure),
                    mime=mime,
                    use_container_width=True,
                    key="btn_dl_image"
                )
    with col_dl2:
        rapport_json = json.dumps(rapport_final, indent=4, ensure_ascii=False)
        st.download_button(
            label="⬇️ Rapport JSON complet",
            data=rapport_json.encode("utf-8"),
            file_name=f"rapport_{st.session_state['nom_image'].split('.')[0]}.json",
            mime="application/json",
            use_container_width=True,
            key="btn_dl_json"
        )

    # ── Features extraites ────────────────────────────────────
    if afficher_details and isinstance(rapport_analyse, dict):
        st.divider()
        st.subheader("🔬 Features extraites (Agent 1)")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("**Complexité visuelle**")
            complexite = rapport_analyse.get("complexite", {})
            st.metric("Entropie",         complexite.get("entropie", 0))
            st.metric("Score complexité", complexite.get("score_complexite", 0))
            st.metric("Niveau",           complexite.get("niveau_complexite", "").upper())
        with col_f2:
            st.markdown("**Textures GLCM**")
            textures = rapport_analyse.get("textures", {})
            st.metric("Contraste",   textures.get("contraste",   0))
            st.metric("Homogénéité", textures.get("homogeneite", 0))
            st.metric("Corrélation", textures.get("correlation", 0))
        with col_f3:
            st.markdown("**OCR**")
            ocr = rapport_analyse.get("ocr", {})
            st.metric("Texte détecté", "✅ Oui" if ocr.get("texte_detecte") else "❌ Non")
            st.metric("Nb mots",        ocr.get("nb_mots", 0))
            st.metric("Confiance OCR", f"{ocr.get('confiance_moyenne', 0)}%")

    # ── Rapport JSON ──────────────────────────────────────────
    if afficher_json:
        st.divider()
        st.subheader("📄 Rapport JSON complet")
        st.json(rapport_final)

# ── Message d'accueil ─────────────────────────────────────────
elif fichier_upload is None:
    st.info("👆 Uploadez une image pour commencer l'analyse intelligente")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 🔍 Agent 1\n**Analyseur**\nFeatures, GLCM, OCR")
    with col2:
        st.markdown("### 🤖 Agent 2\n**Classifier**\nVote 3 LLMs")
    with col3:
        st.markdown("### ⚙️ Agent 3\n**Compresseur**\nJPEG, PNG, WebP, HEIF, AVIF")
    with col4:
        st.markdown("### 📊 Agents 4+5\n**Evaluateur**\nPSNR, SSIM, rapport final")