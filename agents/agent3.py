import os
import cv2
import json
import time
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor

# Activation du support pour les formats modernes (AVIF, HEIF)
register_heif_opener()

class Agent3Ultra:
    def __init__(self, input_dir, output_dir, quality_threshold=0.88, max_workers=4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold = quality_threshold  # Seuil SSIM pour valider la qualité
        self.max_workers = max_workers
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _normalize_and_clean(self, pil_img):
        """Étape 1 : Correction orientation et suppression totale des métadonnées EXIF."""
        img = ImageOps.exif_transpose(pil_img)
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        return clean

    def _classify_vision(self, cv_img):
        """Étape 2 : Analyse par Vision Artificielle (Contours et Variance)."""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        variance = np.var(cv_img)
        
        # Un logo/graphique a des bords nets et peu de dégradés
        if edge_density > 0.05 or variance < 1200:
            return "GRAPHIC_LOGO"
        return "NATURAL_PHOTO"

    def _calculate_ssim(self, original_pil, compressed_path):
        """Étape 3 : Mesure de la ressemblance structurelle (Qualité Perceptive)."""
        comp_img = Image.open(compressed_path).convert('L')
        org_img = original_pil.convert('L').resize(comp_img.size)
        score, _ = ssim(np.array(org_img), np.array(comp_img), full=True)
        return score

    def optimize_image(self, file_path):
        """Cœur de l'Agent 3 : La compétition Multi-Format."""
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        
        try:
            pil_img = Image.open(file_path)
            cv_img = cv2.imread(file_path)
            original_size = os.path.getsize(file_path)

            # 1. Prétraitement
            clean_img = self._normalize_and_clean(pil_img)
            category = self._classify_vision(cv_img)
            
            # 2. Test des formats (La Compétition)
            # On teste AVIF, WebP et JPEG pour trouver le meilleur ratio
            formats_to_test = ['avif', 'webp', 'jpeg']
            best_res = {"path": None, "size": original_size, "fmt": "ORIGINAL", "ssim": 1.0}

            for fmt in formats_to_test:
                temp_path = os.path.join(self.output_dir, f"temp_{name_only}.{fmt}")
                
                # Qualité adaptative selon la catégorie
                q = 82 if category == "NATURAL_PHOTO" else 94
                
                if fmt == 'jpeg':
                    clean_img.convert('RGB').save(temp_path, "JPEG", quality=q, optimize=True)
                else:
                    clean_img.save(temp_path, fmt.upper(), quality=q)

                # Analyse du résultat
                current_ssim = self._calculate_ssim(clean_img, temp_path)
                current_size = os.getsize(temp_path)

                # Décision : On garde si c'est plus léger ET que la qualité SSIM est respectée
                if current_ssim >= self.threshold and current_size < best_res["size"]:
                    if best_res["path"] and os.path.exists(best_res["path"]):
                        os.remove(best_res["path"]) # Nettoyage de l'ancien format testé
                    
                    final_path = os.path.join(self.output_dir, f"{name_only}_opt.{fmt}")
                    os.rename(temp_path, final_path)
                    best_res.update({"path": final_path, "size": current_size, "fmt": fmt, "ssim": current_ssim})
                else:
                    if os.path.exists(temp_path): os.remove(temp_path)

            # Gain final
            gain = round((1 - best_res["size"] / original_size) * 100, 2)
            return {
                "nom": filename,
                "type_detecte": category,
                "format_gagnant": best_res["fmt"],
                "gain_poids": f"{gain}%",
                "qualite_ssim": round(best_res["ssim"], 4)
            }
        except Exception as e:
            return {"nom": filename, "erreur": str(e)}

    def run(self):
        """Traitement de la base de données (30 images)."""
        print(f"--- [AGENT 3] Analyse et compression de la base de données ---")
        # Récupère les 30 premières images valides
        all_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        files_to_process = all_files[:30]
        
        start_time = time.time()
        
        # Parallélisation pour exploiter le processeur à 100%
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.optimize_image, files_to_process))
            
        # Génération du rapport final pour l'Agent 4
        report_path = os.path.join(self.output_dir, 'agent3_final_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        total_time = round(time.time() - start_time, 2)
        print(f"--- [TERMINE] 30 images traitées en {total_time}s ---")
        print(f"--- Rapport généré : {report_path} ---")

if __name__ == "__main__":
    # CONFIGURATION DES CHEMINS
    DOSSIER_SOURCE = "./images_entree"  # Ton dossier avec les 30 images
    DOSSIER_SORTIE = "./resultats_agent3"
    
    if not os.path.exists(DOSSIER_SOURCE):
        os.makedirs(DOSSIER_SOURCE)
        print(f"INFO : Dossier '{DOSSIER_SOURCE}' créé. Place tes 30 images dedans.")
    else:
        # Lancement de l'agent
        mon_agent = Agent3Ultra(DOSSIER_SOURCE, DOSSIER_SORTIE)
        mon_agent.run()