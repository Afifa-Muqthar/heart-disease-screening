"""Flask application for the five-input heart screening estimate."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
STATIC_DIR = PUBLIC_DIR / "static"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "heart_disease_set_b_final.joblib"
CONFIG_PATH = MODEL_DIR / "heart_disease_set_b_final.json"

MODEL = joblib.load(MODEL_PATH)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
FEATURES = CONFIG["feature_order"]

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 4096

# Do not emit request access logs; prediction payloads are never logged by this app.
logging.getLogger("werkzeug").disabled = True


def choose_band(score: float) -> tuple[str, dict]:
    bands = CONFIG["bands"]
    if score <= bands["lower"]["maximum_inclusive"]:
        return "lower", bands["lower"]
    if score < bands["higher"]["minimum_inclusive"]:
        return "intermediate", bands["intermediate"]
    return "higher", bands["higher"]


@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/static/<path:filename>")
def local_static(filename: str):
    """Serve Vercel's public assets during the local Flask run."""
    return send_from_directory(STATIC_DIR, filename)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="The request is too large."), 413


@app.post("/api/predict")
def predict():
    if not request.is_json:
        return jsonify(error="Submit the five inputs as JSON."), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="The submitted values could not be read. Check each input and try again."), 400

    missing = [name for name in FEATURES if name not in payload or payload[name] in (None, "")]
    if missing:
        return jsonify(error="Complete all five inputs before estimating.", missing=missing), 400

    try:
        age = int(payload["age"])
        bp = float(payload["trestbps"])
        if not math.isfinite(bp):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        return jsonify(error="Age and blood pressure must be valid numbers."), 400

    if not 18 <= age <= 100:
        return jsonify(error="Enter an age from 18 to 100."), 400
    if not 80 <= bp <= 200:
        return jsonify(error="Resting systolic blood pressure must be between 80 and 200 mmHg."), 400
    if not 28 <= age <= 77:
        return jsonify(
            band="unavailable",
            band_name="Estimate unavailable",
            copy="This age is outside the dataset range (28–77). The estimate is unreliable, so no score was produced.",
        )
    if payload["sex"] not in CONFIG["allowed_values"]["sex"]:
        return jsonify(error="Choose Female or Male for sex."), 400
    if payload["cp"] not in CONFIG["allowed_values"]["cp"]:
        return jsonify(error="Choose one of the listed chest-pain categories."), 400
    if payload["exang"] not in (True, False):
        return jsonify(error="Choose Yes or No for exertion-related discomfort."), 400

    row = pd.DataFrame([{key: payload[key] for key in FEATURES}], columns=FEATURES)
    row["age"] = age
    row["trestbps"] = bp
    score = float(MODEL.predict_proba(row)[0, 1])
    band_key, band = choose_band(score)
    # Keep the public response categorical. The reference line remains display copy.
    copy = f"{band['description']}\n{band['reference_line']}"
    return jsonify(band=band_key, band_name=band["name"], copy=copy)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    deployed = os.environ.get("APP_ENV", "development").lower() == "production"
    host = "0.0.0.0" if deployed else "127.0.0.1"
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
