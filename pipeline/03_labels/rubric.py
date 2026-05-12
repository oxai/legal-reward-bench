from __future__ import annotations

from typing import Any

RUBRIC_VERSION = "answer_behavior_v1+semantic_labels_v1"

FINAL_LABEL_VALUES = {
    "answer_behavior": {"attempted", "abstained", "unusable"},
    "faithfulness": {
        "fully_supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "not_applicable",
    },
    "correctness": {"correct", "incorrect", "not_applicable"},
    "completeness": {"complete", "incomplete", "not_applicable"},
    "conciseness": {"concise", "acceptable", "not_concise", "not_applicable"},
}

BEHAVIOR_VALUES = {"answer_behavior": FINAL_LABEL_VALUES["answer_behavior"]}
SEMANTIC_VALUES = {
    key: values - {"not_applicable"} for key, values in FINAL_LABEL_VALUES.items() if key != "answer_behavior"
}
REQUIRED_FINAL_LABELS = tuple(FINAL_LABEL_VALUES)
DOWNSTREAM_LABELS = REQUIRED_FINAL_LABELS[1:]


def validate_behavior_payload(label: dict[str, Any]) -> list[str]:
    return validate_keys_and_values(label, BEHAVIOR_VALUES)


def validate_semantic_payload(label: dict[str, Any]) -> list[str]:
    return validate_keys_and_values(label, SEMANTIC_VALUES)


def validate_final_label_payload(label: dict[str, Any]) -> list[str]:
    errors = validate_keys_and_values(label, FINAL_LABEL_VALUES)

    answer_behavior = label.get("answer_behavior")
    if answer_behavior == "attempted":
        for key in DOWNSTREAM_LABELS:
            if label.get(key) == "not_applicable":
                errors.append(f"{key} must not be 'not_applicable' for attempted answers")
    elif answer_behavior in {"abstained", "unusable"}:
        for key in DOWNSTREAM_LABELS:
            if label.get(key) != "not_applicable":
                errors.append(f"{key} must be 'not_applicable' for {answer_behavior} responses")

    return errors


def validate_keys_and_values(label: dict[str, Any], allowed_values: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []
    if set(label) != set(allowed_values):
        errors.append(f"payload must contain exactly these keys: {list(allowed_values)}")

    for key, values in allowed_values.items():
        value = label.get(key)
        if value not in values:
            errors.append(f"{key} must be one of {sorted(values)}, got {value!r}")

    return errors
