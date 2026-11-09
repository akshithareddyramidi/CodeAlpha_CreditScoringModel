"""Append privacy-conscious prediction activity records to a local CSV file."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

AUDIT_COLUMNS = [
    "prediction_id",
    "timestamp_utc",
    "selected_model",
    "predicted_label",
    "probability_creditworthy",
    "all_model_outputs_json",
    "dataset_type",
    "app_version",
]


def append_audit_record(
    log_path: Path,
    *,
    selected_model: str,
    predicted_label: str,
    probability_creditworthy: float,
    all_model_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Append one prediction event without storing names or raw financial inputs.

    The log records prediction metadata and model outputs only. It intentionally
    does not write applicant names, ages, income, debt, payment history, or other
    submitted form values.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "prediction_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": selected_model,
        "predicted_label": predicted_label,
        "probability_creditworthy": round(float(probability_creditworthy), 6),
        "all_model_outputs_json": json.dumps(all_model_outputs, sort_keys=True),
        "dataset_type": "synthetic_teaching_data",
        "app_version": "1.1.0",
    }

    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return record
