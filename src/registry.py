"""Central disease registry loaded from configuration.

Application code should query this registry rather than hard-coding disease
lists. Adding a disease is a configuration change; it does not train a model
or change clinical meaning until data, training, and evaluation are updated.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from paths import DEFAULT_DISEASES_FILE


@dataclass(frozen=True)
class Disease:
    id: str
    display_name: str
    enabled: bool

    @classmethod
    def from_dict(cls, data: dict) -> Disease:
        if not isinstance(data, dict):
            raise ValueError("Each disease record must be an object.")

        disease_id = data.get("id")
        display_name = data.get("display_name")
        enabled = data.get("enabled")

        if not isinstance(disease_id, str) or not disease_id.strip():
            raise ValueError("Each disease must have a non-empty string id.")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"Disease '{disease_id}' must have a non-empty display_name.")
        if not isinstance(enabled, bool):
            raise ValueError(f"Disease '{disease_id}' enabled must be a boolean.")

        return cls(
            id=disease_id.strip(),
            display_name=display_name.strip(),
            enabled=enabled,
        )


class DiseaseRegistry:
    """In-memory view of the configured disease set (N labels)."""

    def __init__(self, diseases: Sequence[Disease]):
        ids = [disease.id for disease in diseases]
        duplicates = sorted(disease_id for disease_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate disease ids are not allowed: {duplicates}")

        self._diseases = {disease.id: disease for disease in diseases}

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> DiseaseRegistry:
        return cls([Disease.from_dict(record) for record in records])

    @classmethod
    def from_json(cls, path: Path | None = None) -> DiseaseRegistry:
        registry_path = path or DEFAULT_DISEASES_FILE
        with registry_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict) or "diseases" not in payload:
            raise ValueError("diseases.json must be an object with a 'diseases' list.")

        records = payload["diseases"]
        if not isinstance(records, list):
            raise ValueError("diseases.json 'diseases' must be a list.")

        return cls.from_records(records)

    def all(self) -> list[Disease]:
        return list(self._diseases.values())

    def enabled(self) -> list[Disease]:
        return [disease for disease in self._diseases.values() if disease.enabled]

    def ids(self, *, enabled_only: bool = False) -> list[str]:
        diseases = self.enabled() if enabled_only else self.all()
        return [disease.id for disease in diseases]

    def get(self, disease_id: str) -> Disease:
        try:
            return self._diseases[disease_id]
        except KeyError as exc:
            raise KeyError(f"Unknown disease id: {disease_id}") from exc

    def contains(self, disease_id: str) -> bool:
        return disease_id in self._diseases

    def is_enabled(self, disease_id: str) -> bool:
        return self.get(disease_id).enabled

    def __len__(self) -> int:
        return len(self._diseases)
