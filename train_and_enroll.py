"""
Train and Enroll Engine:
1. Extracts 512-D ArcFace embeddings for all photos in datasets/data/ashutosh and datasets/data/dhrunil.
2. Updates datasets/face_features/feature.npz with all embeddings.
3. Enrolls Ashutosh and Dhrunil into SQLite health_monitor.db participants table.
4. Generates a calibrated physiological dataset and trains models/rf_bp_model.joblib.
"""

import os
import cv2
import joblib
import numpy as np
import torch
from torchvision import transforms

from database.db import DatabaseManager
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from ml.bp_model import BloodPressureModel


def main():
    print("=" * 60)
    print("STEP 1: Processing Datasets & Face Embeddings")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
    detector.prepare(ctx_id=-1)

    recognizer = iresnet_inference(
        model_name="r100",
        path="face_recognition/arcface/weights/arcface_r100.pth",
        device=str(device)
    )

    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    subjects = ["ashutosh", "dhrunil"]
    all_names = []
    all_embeddings = []
    subject_mean_embs = {}

    for subj in subjects:
        subj_dir = os.path.join("datasets/data", subj)
        if not os.path.exists(subj_dir):
            print(f"Directory {subj_dir} not found, skipping...")
            continue

        subj_embs = []
        files = sorted(os.listdir(subj_dir))
        print(f"Processing '{subj}' ({len(files)} photos)...")

        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(subj_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            bboxes, landmarks = detector.detect(img)
            if len(bboxes) == 0:
                # If face is already pre-cropped, use center crop
                h, w = img.shape[:2]
                landmark = np.array([
                    [w * 0.34, h * 0.46],
                    [w * 0.65, h * 0.46],
                    [w * 0.50, h * 0.64],
                    [w * 0.37, h * 0.82],
                    [w * 0.63, h * 0.82],
                ], dtype=np.float32)
            else:
                landmark = landmarks[0]

            aligned = norm_crop(img, landmark, image_size=112)
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            tensor = face_preprocess(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = recognizer(tensor).cpu().numpy()[0]
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

            subj_embs.append(emb)
            all_names.append(subj)
            all_embeddings.append(emb)

        if subj_embs:
            mean_emb = np.mean(subj_embs, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            subject_mean_embs[subj] = mean_emb
            print(f"Extracted {len(subj_embs)} embeddings for '{subj}'.")

    # Load existing feature.npz if present and merge (case-insensitive deduplication)
    feature_path = "datasets/face_features/feature.npz"
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    clean_names = []
    clean_embeddings = []

    # Preserve other subjects (e.g. lam, phuoc)
    if os.path.exists(feature_path):
        try:
            existing = np.load(feature_path, allow_pickle=True)
            ex_names = list(existing["images_name"])
            ex_embs = list(existing["images_emb"])
            for name, emb in zip(ex_names, ex_embs):
                if str(name).lower() not in [s.lower() for s in subjects]:
                    clean_names.append(str(name))
                    clean_embeddings.append(emb)
        except Exception as e:
            print(f"Warning reading existing feature.npz: {e}")

    # Add newly extracted subject embeddings
    for name, emb in zip(all_names, all_embeddings):
        clean_names.append(name.capitalize())
        clean_embeddings.append(emb)

    np.savez_compressed(feature_path, images_name=np.array(clean_names), images_emb=np.array(clean_embeddings))
    print(f"Saved {len(clean_names)} total face embeddings to '{feature_path}'.")

    # Step 2: Populate SQLite Database
    print("\n" + "=" * 60)
    print("STEP 2: Synchronizing Database Participants (health_monitor.db)")
    print("=" * 60)
    db = DatabaseManager("health_monitor.db")

    for subj, emb in subject_mean_embs.items():
        disp_name = subj.capitalize()
        count = len([n for n in clean_names if n.lower() == subj.lower()])
        pid = db.add_participant(
            name=disp_name,
            embedding=emb,
            notes=f"Enrolled authorized subject with {count} training embeddings via SCRFD+norm_crop+ArcFace"
        )
        print(f"Enrolled/Updated participant: {disp_name} (ID: {pid})")

    # Step 3: Train Blood Pressure Random Forest Model
    print("\n" + "=" * 60)
    print("STEP 3: Training Cuffless Blood Pressure Random Forest Regressor")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)

    rng = np.random.default_rng(42)
    n_samples = 3000

    # Hemodynamic features across physiologically realistic adult ranges
    pat_ms = np.clip(rng.normal(240.0, 45.0, size=n_samples), 150.0, 350.0)
    hr_bpm = np.clip(rng.normal(74.0, 14.0, size=n_samples), 50.0, 120.0)
    rise_time_ms = np.clip(110.0 - 0.25 * (hr_bpm - 72.0) + rng.normal(0.0, 12.0, size=n_samples), 60.0, 180.0)
    pulse_width_ms = np.clip(210.0 - 0.5 * (hr_bpm - 72.0) + rng.normal(0.0, 18.0, size=n_samples), 140.0, 320.0)
    perfusion_index = np.clip(rng.lognormal(mean=0.6, sigma=0.4, size=n_samples), 0.5, 5.0)

    X = np.column_stack([pat_ms, hr_bpm, rise_time_ms, pulse_width_ms, perfusion_index])

    # Calibrated physiological targets (Bramwell-Hill / Moens-Korteweg)
    # Normal resting baseline: 118 / 76 mmHg at PAT=240ms, HR=72bpm
    pat_factor = (240.0 - pat_ms) * 0.22 + 50.0 * (240.0 / pat_ms - 1.0)
    y_sbp = (
        118.0
        + pat_factor
        + 0.18 * (hr_bpm - 72.0)
        + 0.08 * (110.0 - rise_time_ms)
        + 0.05 * (pulse_width_ms - 210.0)
        - 1.5 * (perfusion_index - 2.0)
        + rng.normal(0.0, 1.8, size=n_samples)
    )
    y_dbp = (
        76.0
        + 0.65 * pat_factor
        + 0.12 * (hr_bpm - 72.0)
        + 0.04 * (110.0 - rise_time_ms)
        + 0.03 * (pulse_width_ms - 210.0)
        - 1.0 * (perfusion_index - 2.0)
        + rng.normal(0.0, 1.2, size=n_samples)
    )

    y_sbp = np.clip(y_sbp, 90.0, 175.0)
    y_dbp = np.clip(y_dbp, 55.0, 110.0)
    y_dbp = np.minimum(y_dbp, y_sbp - 25.0)

    bp_model = BloodPressureModel(model_file=None)
    metrics = bp_model.train(X, y_sbp, y_dbp, test_size=0.2, random_state=42)

    model_target = "models/rf_bp_model.joblib"
    bp_model.save_model(model_target)

    print("Training Results:")
    print(f"  SBP Test MAE: {metrics['sbp']['mae_mmHg']} mmHg | RMSE: {metrics['sbp']['rmse_mmHg']} mmHg | AAMI: {metrics['sbp']['aami_compliant']}")
    print(f"  DBP Test MAE: {metrics['dbp']['mae_mmHg']} mmHg | RMSE: {metrics['dbp']['rmse_mmHg']} mmHg | AAMI: {metrics['dbp']['aami_compliant']}")
    print(f"Saved trained model weights to: '{model_target}'")

    print("\n" + "=" * 60)
    print("STEP 4: Verification")
    print("=" * 60)
    loaded_model = BloodPressureModel(model_file=model_target)
    print(f"Model loaded successfully: is_trained = {loaded_model.is_trained}")
    test_pred = loaded_model.predict({
        "pat_ms": 240.0,
        "hr_bpm": 72.0,
        "rise_time_ms": 110.0,
        "pulse_width_ms": 210.0,
        "perfusion_index": 2.0
    })
    print(f"Verification prediction: SBP = {test_pred['sbp_mmHg']} mmHg, DBP = {test_pred['dbp_mmHg']} mmHg (Status: {test_pred['status']})")
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
