"""Persistent JSON-lines inference worker for the EV battery passport API."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "files(1)"
sys.path.insert(0, str(MODEL_DIR))

from ml_model import CONFIDENCE_THRESHOLD, FEATURES, evaluate_battery  # noqa: E402


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def build_features(telemetry: dict, capacity_kwh: float) -> pd.DataFrame:
    """Map available pack telemetry to the model's cell-level feature schema.

    Some model inputs are proxies because the demo's BMS feed does not provide
    cell-level dispersion, resistance, charge time, or discharge capacity.
    """
    soh = clamp(telemetry.get("stateOfHealth", 90), 0, 100)
    current = abs(float(telemetry.get("current", 0)))
    c_rate = clamp(current / 12.5, 0.5, 2.0)
    current_mean = c_rate * 2.2
    temperature_mean = float(telemetry.get("temperatureC", 25))
    ambient_temp = clamp(temperature_mean - current_mean * 1.8, 20, 40)
    pack_voltage = float(telemetry.get("voltage", 365))
    estimated_series_cells = max(1, round(pack_voltage / 3.65))
    voltage_mean = pack_voltage / estimated_series_cells

    return pd.DataFrame([{
        "cycle_number": int(telemetry.get("cycleCount", 0)),
        "ambient_temp_c": ambient_temp,
        "c_rate": c_rate,
        "voltage_mean": voltage_mean,
        "voltage_std": 0.05 + (100 - soh) * 0.0008,
        "current_mean": current_mean,
        "temperature_mean": temperature_mean,
        "internal_resistance_mohm": 20 + (100 - soh) * 0.9,
        "charge_time_min": 55 + (100 - soh) * 0.35,
        "discharge_capacity_ah": 2.0 * (soh / 100.0),
    }], columns=FEATURES)


def main() -> None:
    soh_model = joblib.load(MODEL_DIR / "soh_model.joblib")
    rul_model = joblib.load(MODEL_DIR / "rul_model.joblib")
    sys.stdout.write(json.dumps({"type": "ready", "status": "ready", "model": "battery-soh-rul-random-forest"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            features = build_features(request["telemetry"], float(request.get("capacityKwh", 0)))
            event = evaluate_battery(request["batteryId"], features, soh_model, rul_model)
            result = {
                "requestId": request["requestId"],
                "type": "prediction",
                "prediction": {
                    "batteryId": event.battery_id,
                    "telemetryId": request["telemetry"].get("id"),
                    "cycleNumber": event.cycle_number,
                    "predictedSoh": event.predicted_soh,
                    "sohConfidence": event.soh_confidence,
                    "predictedRul": event.predicted_rul,
                    "rulConfidence": event.rul_confidence,
                    "compatibility": event.compatibility,
                    "overallConfidence": event.overall_confidence,
                    "confidenceThreshold": CONFIDENCE_THRESHOLD,
                    "decision": event.decision,
                    "timestamp": request["telemetry"].get("timestamp"),
                    "model": "battery-soh-rul-random-forest",
                    "trainingData": "synthetic battery-cycle dataset",
                    "inputFeaturesAreProxies": True,
                    "capacityKwh": request.get("capacityKwh"),
                },
            }
        except Exception as error:  # keep the worker alive for subsequent readings
            result = {
                "requestId": request.get("requestId") if "request" in locals() else None,
                "type": "error",
                "error": str(error),
            }
            traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(json.dumps({"type": "startup_error", "error": str(error)}) + "\n")
        sys.stdout.flush()
        raise
