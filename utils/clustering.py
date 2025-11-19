import joblib
import numpy as np

# Load saved model
centroids = joblib.load("model/centroids.pkl")
scaler = joblib.load("model/scaler.pkl")

def assign_cluster(new_data):
    """Assign cluster index based on nearest centroid"""
    new_data_scaled = scaler.transform([new_data])
    distances = np.linalg.norm(new_data_scaled - centroids, axis=1)
    return int(np.argmin(distances))
