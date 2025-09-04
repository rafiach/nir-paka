from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import pickle
import os
import json

# --------------------------
# Load model, scaler, dan metadata saat server start
# --------------------------
BASE_DIR = os.path.dirname(__file__)
artifact_dir = os.path.join(BASE_DIR, "../")

with open(os.path.join(artifact_dir, "rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(artifact_dir, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(artifact_dir, "metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
target_cols = meta["target_cols"]
USE_SNV = meta["use_snv"]

# --------------------------
# SNV transform manual
# --------------------------
def snv_transform(X_array):
    X = np.asarray(X_array, dtype=float)
    row_mean = X.mean(axis=1, keepdims=True)
    row_std  = X.std(axis=1, keepdims=True) + 1e-8
    return (X - row_mean) / row_std

# --------------------------
# API Endpoint
# --------------------------
@api_view(["POST"])
def predict(request):
    try:
        data = request.data.get("inputs", [])
        if len(data) != len(feature_cols):
            return Response(
                {"error": f"Butuh {len(feature_cols)} parameter sesuai urutan {feature_cols}"},
                status=400
            )

        # Convert input ke array 2D
        X = np.array(data, dtype=float).reshape(1, -1)

        # Preprocessing manual
        if USE_SNV:
            X = snv_transform(X)
        X = scaler.transform(X)

        # Prediksi multi-target
        y_pred = model.predict(X)[0]

        # Buat response dict
        result = {target_cols[i]: float(y_pred[i]) for i in range(len(target_cols))}
        return Response({"prediction": result})

    except Exception as e:
        return Response({"error": str(e)}, status=500)
