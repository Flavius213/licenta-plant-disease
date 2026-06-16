from __future__ import annotations

import json
from pathlib import Path

from src.config import BASE_DIR


RULES_PATH = BASE_DIR / "ontology" / "plant_rules.json"


def load_rules(path: Path = RULES_PATH) -> dict[str, dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def rule_for_class(class_name: str, rules: dict[str, dict[str, object]] | None = None) -> dict[str, object] | None:
    rules = rules or load_rules()
    return rules.get(class_name)


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value:
        return [str(value)]
    return []


def _symbolic_state(rule: dict[str, object]) -> str:
    status = str(rule.get("status", "")).lower()
    health_label = str(rule.get("health_label", "")).strip()
    if status == "healthy":
        return health_label or "healthy"
    if status == "diseased":
        return health_label or "diseased"
    return health_label or "unknown"


def format_rule(class_name: str, probability: float, rule: dict[str, object] | None) -> str:
    confidence_percent = probability * 100
    lines = [
        f"Prediction: {class_name} ({confidence_percent:.2f}% confidence)",
    ]

    if not rule:
        lines.extend(
            [
                "Detected leaf: unknown",
                "Symbolic state: no rule",
                "No symbolic rules are available yet for this class.",
            ]
        )
        return "\n".join(lines)

    plant = str(rule.get("plant", "unknown"))
    leaf_type = str(rule.get("leaf_type") or f"{plant} leaf")
    diagnosis = str(rule.get("diagnosis", "unknown diagnosis"))
    likely_cause = str(rule.get("likely_cause", "unknown cause"))
    symbolic_summary = str(rule.get("symbolic_summary", "")).strip()

    lines.extend(
        [
            f"Detected leaf: {leaf_type}",
            f"Symbolic state: {_symbolic_state(rule)}",
            f"Diagnosis: {diagnosis}",
            f"Plant: {plant}",
            f"Likely cause: {likely_cause}",
            "",
        ]
    )

    if symbolic_summary:
        lines.extend(["Symbolic interpretation:", f"- {symbolic_summary}", ""])

    lines.append("Observed symptoms:")
    lines.extend(f"- {symptom}" for symptom in _as_list(rule.get("visual_symptoms")))
    lines.append("")
    lines.append("Soil recommendation:")
    lines.extend(f"- {item}" for item in _as_list(rule.get("soil_recommendation")))
    lines.append("")
    lines.append("Recommended actions:")
    lines.extend(f"- {item}" for item in _as_list(rule.get("treatment")))

    if probability < 0.75:
        lines.extend(
            [
                "",
                "Warning: confidence is below 75%, so manually verify the image or take a clearer photo.",
            ]
        )

    return "\n".join(lines)
