"""
Script to process raw face images, extract 512-dim ArcFace embeddings with norm_crop alignment,
update datasets/face_features/feature.npz, and enroll Ashutosh and Dhrunil into health_monitor.db.
"""

import os
import sys

# Ensure repository root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import torch
from torchvision import transforms

from database.db import DatabaseManager
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_pipeline.face_service import FaceService
from face_recognition.arcface.model import iresnet_inference


def main():
    print("=== Step 1 & 2: Extracting ArcFace embeddings with norm_crop ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
    detector.prepare(ctx_id=-1)

    recognizer = iresnet_inference(
        model_name="r100",
        path="face_recognition/arcface/weights/arcface_r100.pth",
        device=str(device),
    )

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    @torch.no_grad()
    def extract_embedding(aligned_face: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        tensor = preprocess(rgb).unsqueeze(0).to(device)
        raw = recognizer(tensor)[0].cpu().numpy()
        norm = np.linalg.norm(raw)
        return raw / norm if norm > 0 else raw

    # Read existing feature.npz if present to preserve other persons (lam, phuoc)
    existing_names = []
    existing_embs = []
    feat_path = "datasets/face_features/feature.npz"
    if os.path.exists(feat_path):
        data = np.load(feat_path, allow_pickle=True)
        old_names = list(data["images_name"])
        old_embs = list(data["images_emb"])
        for n, e in zip(old_names, old_embs):
            # Exclude prior versions of ashutosh/dhrunil to cleanly update them
            if n.lower() not in ["ashutosh", "dhrunil"]:
                existing_names.append(n)
                existing_embs.append(e)
        print(f"Preserved {len(existing_names)} existing entries: {set(existing_names)}")

    # We will process all target people
    people_config = [
        ("Ashutosh", "datasets/data/ashutosh"),
        ("Dhrunil", "datasets/data/dhrunil"),
    ]

    all_names = list(existing_names)
    all_embs = list(existing_embs)
    person_embeddings = {}

    for display_name, folder in people_config:
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue

        files = sorted(
            [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        print(f"\nProcessing {display_name} in {folder} ({len(files)} images)...")
        extracted_embs = []

        for f in files:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: could not read {img_path}")
                continue

            bboxes, landmarks = detector.detect(img)
            if len(bboxes) == 0:
                print(f"  Warning: no face detected in {f}")
                continue

            # Select face with highest detection score
            best_idx = int(np.argmax(bboxes[:, 4]))
            best_box = bboxes[best_idx]
            best_lmk = landmarks[best_idx]

            # Landmark-guided norm_crop alignment
            aligned_face = norm_crop(img, best_lmk, image_size=112)
            emb = extract_embedding(aligned_face)

            extracted_embs.append(emb)
            all_names.append(display_name)
            all_embs.append(emb)
            print(f"  [OK] {f}: box score={best_box[4]:.2f}, norm_crop 112x112 -> 512-d emb")

        person_embeddings[display_name] = np.array(extracted_embs)
        print(f"Total extracted for {display_name}: {len(extracted_embs)} valid faces")

    # Step 3: Save/update datasets/face_features/feature.npz
    print("\n=== Step 3: Saving updated feature.npz ===")
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    images_name_arr = np.array(all_names)
    images_emb_arr = np.array(all_embs)
    np.savez_compressed(
        feat_path,
        images_name=images_name_arr,
        images_emb=images_emb_arr,
    )
    print(f"Successfully saved {feat_path}:")
    print(f"  images_name count: {len(images_name_arr)}")
    print(f"  images_emb shape: {images_emb_arr.shape}")
    print(f"  Unique identities: {sorted(list(set(images_name_arr)))}")

    # Step 4: Synchronize into health_monitor.db participants table
    print("\n=== Step 4: Synchronizing SQLite health_monitor.db ===")
    db = DatabaseManager("health_monitor.db")

    for display_name in ["Ashutosh", "Dhrunil"]:
        embs = person_embeddings.get(display_name)
        if embs is None or len(embs) == 0:
            print(f"No embeddings found for {display_name}")
            continue

        # Compute unit-normalized averaged embedding
        mean_emb = np.mean(embs, axis=0)
        norm_mean = mean_emb / np.linalg.norm(mean_emb)

        # Calculate consistency metrics
        similarities = np.dot(embs, norm_mean)
        notes = (
            f"Enrolled via SCRFD+norm_crop+ArcFace | {len(embs)} images | "
            f"CosSim Mean: {similarities.mean():.3f}, Min: {similarities.min():.3f}, Max: {similarities.max():.3f}"
        )

        p_id = db.add_participant(
            name=display_name,
            embedding=norm_mean,
            notes=notes,
        )
        print(f"Enrolled '{display_name}' (ID: {p_id}) into health_monitor.db")
        print(f"  Notes: {notes}")

    # Verify FaceService recognizes both out-of-the-box
    print("\n=== Verifying FaceService In-Memory Cache and Recognition ===")
    face_srv = FaceService(db)
    print(f"FaceService is_ready: {face_srv.is_ready}")
    print(f"Known names in FaceService: {face_srv.known_names}")
    print(f"Known embeddings shape: {face_srv.known_embeddings.shape if face_srv.known_embeddings is not None else None}")
    print(f"Participant ID mapping: {face_srv.participant_id_map}")

    # Test sample recognition on Ashutosh and Dhrunil
    test_samples = [
        ("Ashutosh", "datasets/data/ashutosh/0.jpg"),
        ("Dhrunil", "datasets/data/dhrunil/0.jpg"),
    ]
    for expected_name, test_img_path in test_samples:
        test_img = cv2.imread(test_img_path)
        res = face_srv.process_frame(test_img)
        detections = res.get("detections", [])
        if detections:
            det = detections[0]
            print(f"Test image {test_img_path}:")
            print(f"  Recognized Name: {det['name']} (Expected: {expected_name})")
            print(f"  Similarity: {det['similarity']}")
            print(f"  Is Known: {det['is_known']}")
            print(f"  Participant ID: {det['participant_id']}")
            assert det["name"] == expected_name, f"Expected {expected_name} but got {det['name']}"
            assert det["similarity"] >= 0.70, f"Similarity {det['similarity']} below 0.70 threshold"
        else:
            print(f"Warning: No detection in test image {test_img_path}")

    print("\n=== Face processing, feature.npz update, and DB enrollment completed successfully! ===")


if __name__ == "__main__":
    main()
