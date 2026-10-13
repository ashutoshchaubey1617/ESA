import numpy as np


def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = list(data["images_name"])
        images_emb = list(data["images_emb"])

        return images_name, images_emb
    except Exception:
        return [], []


def compare_encodings(encoding, encodings):
    if encodings is None or len(encodings) == 0:
        return np.array([0.0]), 0
    sims = np.dot(encodings, encoding.T)
    pare_index = int(np.argmax(sims))
    score = sims[pare_index]
    return score, pare_index

