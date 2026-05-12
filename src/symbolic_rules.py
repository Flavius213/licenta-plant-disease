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


def format_rule(class_name: str, probability: float, rule: dict[str, object] | None) -> str:
    confidence_percent = probability * 100
    lines = [
        f"Predictie: {class_name} ({confidence_percent:.2f}% incredere)",
    ]

    if not rule:
        lines.append("Nu exista inca reguli simbolice pentru aceasta clasa.")
        return "\n".join(lines)

    lines.extend(
        [
            f"Diagnostic: {rule['diagnosis']}",
            f"Planta: {rule['plant']}",
            f"Cauza probabila: {rule['likely_cause']}",
            "",
            "Simptome urmarite:",
        ]
    )
    lines.extend(f"- {symptom}" for symptom in rule["visual_symptoms"])
    lines.append("")
    lines.append("Ce pui sau NU pui in sol:")
    lines.extend(f"- {item}" for item in rule["soil_recommendation"])
    lines.append("")
    lines.append("Actiuni recomandate:")
    lines.extend(f"- {item}" for item in rule["treatment"])

    if probability < 0.75:
        lines.extend(
            [
                "",
                "Atentie: increderea este sub 75%, deci verifica manual imaginea sau fa o poza mai clara.",
            ]
        )

    return "\n".join(lines)
