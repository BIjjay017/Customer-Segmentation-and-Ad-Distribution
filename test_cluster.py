# debug_cluster.py
import joblib, numpy as np, pandas as pd, sys
from pathlib import Path

scaler = joblib.load("Model/scaler.pkl")
centroids = joblib.load("Model/centroids.pkl")

print("scaler type:", type(scaler))
print("scaler.feature_names_in_ (if any):", getattr(scaler, "feature_names_in_", None))
print("centroids shape:", centroids.shape)
print("centroids min,max:", centroids.min(), centroids.max())

# Heuristic: centroids in [0,1] -> probably already scaled.
centroids_looks_scaled = (centroids.min() >= -0.1 and centroids.max() <= 1.1)
print("centroids looks scaled?:", centroids_looks_scaled)

# Show scaler data bounds (if MinMaxScaler)
if hasattr(scaler, "data_min_"):
    print("scaler.data_min_ (first 5):", scaler.data_min_[:5])
    print("scaler.data_max_ (first 5):", scaler.data_max_[:5])

# Optional: test a small CSV you pass to this script
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    df = pd.read_csv(path)
    # normalize column names (same as backend)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    FEATURE_ORDER = ["balance","purchases","cash_advance","credit_limit","payments","full_payment","purchases_freq","cash_adv_freq"]
    # Make sure columns exist
    print("have columns:", [c for c in FEATURE_ORDER if c in df.columns])
    X = df[FEATURE_ORDER].astype(float).values
    Xs = scaler.transform(X)            # MUST use transform not fit_transform
    # If centroids are scaled compare Xs to centroids; else scale centroids.
    if centroids_looks_scaled:
        c = centroids
    else:
        c = scaler.transform(centroids)
    dists = np.linalg.norm(Xs[:,None,:] - c[None,:,:], axis=2)
    assigned = dists.argmin(axis=1)
    import collections
    print("cluster counts:", collections.Counter(assigned))
