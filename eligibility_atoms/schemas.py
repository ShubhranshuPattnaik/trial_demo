from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

class SexAllowed(str, Enum):
    male = "male"
    female = "female"
    all = "all"
    unknown = "unknown"

@dataclass
class LabThresholds:
    anc_per_uL_min: Optional[int] = None
    platelets_per_uL_min: Optional[int] = None
    hemoglobin_g_dL_min: Optional[float] = None
    creatinine_mg_dL_max: Optional[float] = None
    ast_uL_max: Optional[float] = None
    alt_uL_max: Optional[float] = None
    bilirubin_mg_dL_max: Optional[float] = None

@dataclass
class TrialAtoms:
    min_age_years: Optional[int] = None
    max_age_years: Optional[int] = None
    sex_allowed: SexAllowed = SexAllowed.unknown
    required_cancer_type: Optional[str] = None
    required_stage: Optional[str] = None
    required_biomarkers: List[str] = field(default_factory=list)
    excluded_biomarkers: List[str] = field(default_factory=list)
    ecog_max: Optional[int] = None
    required_or_allowed_therapies: List[str] = field(default_factory=list)
    disallowed_therapies: List[str] = field(default_factory=list)
    lab_thresholds: LabThresholds = field(default_factory=LabThresholds)
    excluded_conditions: List[str] = field(default_factory=list)
    windows: Dict[str, Optional[int]] = field(default_factory=lambda: {"washout_days": None, "progression_within_days": None})

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sex_allowed"] = self.sex_allowed.value if isinstance(self.sex_allowed, SexAllowed) else self.sex_allowed
        return d

@dataclass
class EligibilityText:
    raw: str
    normalized: str
    inclusion_items: List[str]
    exclusion_items: List[str]

@dataclass
class Provenance:
    # maps field -> list of {value, source, section}
    data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    def add(self, field: str, value: Any, source_line: str, section: str):
        self.data.setdefault(field, []).append({"value": value, "source": source_line.strip(), "section": section})

def json_schema_for_trial_atoms() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "min_age_years": {"type": ["integer","null"], "minimum": 0},
            "max_age_years": {"type": ["integer","null"], "minimum": 0},
            "sex_allowed": {"type": "string", "enum": ["male","female","all","unknown"]},
            "required_cancer_type": {"type": ["string","null"]},
            "required_stage": {"type": ["string","null"]},
            "required_biomarkers": {"type": "array", "items": {"type": "string"}},
            "excluded_biomarkers": {"type": "array", "items": {"type": "string"}},
            "ecog_max": {"type": ["integer","null"], "minimum": 0, "maximum": 4},
            "required_or_allowed_therapies": {"type": "array", "items": {"type": "string"}},
            "disallowed_therapies": {"type": "array", "items": {"type": "string"}},
            "lab_thresholds": {
                "type": "object",
                "properties": {
                    "anc_per_uL_min": {"type": ["integer","null"]},
                    "platelets_per_uL_min": {"type": ["integer","null"]},
                    "hemoglobin_g_dL_min": {"type": ["number","null"]},
                    "creatinine_mg_dL_max": {"type": ["number","null"]},
                    "ast_uL_max": {"type": ["number","null"]},
                    "alt_uL_max": {"type": ["number","null"]},
                    "bilirubin_mg_dL_max": {"type": ["number","null"]},
                },
                "additionalProperties": False
            },
            "excluded_conditions": {"type": "array", "items": {"type": "string"}},
            "windows": {
                "type": "object",
                "properties": {
                    "washout_days": {"type": ["integer","null"]},
                    "progression_within_days": {"type": ["integer","null"]},
                },
                "additionalProperties": False
            },
        },
        "required": [
            "sex_allowed","required_biomarkers","excluded_biomarkers",
            "required_or_allowed_therapies","disallowed_therapies",
            "excluded_conditions","lab_thresholds","windows"
        ],
        "additionalProperties": False
    }
