# debug_distances.py
import joblib, pandas as pd, numpy as np, sys

scaler = joblib.load("Model/scaler.pkl")
centroids = joblib.load("Model/centroids.pkl")

# ordered names used by scaler
FEATURE_NAMES = list(scaler.feature_names_in_)
FEATURE_NAMES_LOWER = [n.strip().lower().replace(" ", "_") for n in FEATURE_NAMES]

def inspect_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in FEATURE_NAMES_LOWER if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}\nExpected: {FEATURE_NAMES_LOWER}")
    X = df[FEATURE_NAMES_LOWER].astype(float)
    X.columns = FEATURE_NAMES  # give scaler the names it expects
    Xs = scaler.transform(X)   # scaled features
    dists = np.linalg.norm(Xs[:, None, :] - centroids[None, :, :], axis=2)  # (n_rows, n_clusters)

    for i, row in df.iterrows():
        scaled = Xs[i].tolist()
        distances = dists[i].tolist()
        best = int(np.argmin(distances))
        best_dist = float(min(distances))
        # second best
        sorted_idx = np.argsort(distances)
        second = int(sorted_idx[1])
        second_dist = float(distances[second])
        gap = second_dist - best_dist
        print(f"\nRow {i} email={row.get('email','')}")
        print("  scaled:", [round(x,4) for x in scaled])
        print("  distances:", [round(x,4) for x in distances])
        print(f"  assigned: {best}  (dist={best_dist:.4f}),  second: {second} (dist={second_dist:.4f}), gap={gap:.4f}")
    print("\nSummary counts:", dict(pd.Series(dists.argmin(axis=1)).value_counts()))
    df["cluster_assigned"] = dists.argmin(axis=1)
    df.to_csv("debug_distances_out.csv", index=False)
    print("Wrote debug_distances_out.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_distances.py your_test.csv")
    else:
        inspect_csv(sys.argv[1])
