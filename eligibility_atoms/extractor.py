from typing import Dict, Any
from .schemas import json_schema_for_trial_atoms
from .normalizer import split_and_normalize
from .rules import extract_with_rules

def extract_atoms_from_eligibility(raw_eligibility_text: str) -> Dict[str, Any]:
    """
    normalize -> split inclusion/exclusion -> rule extraction
    returns a payload ready for LLM function-calling + completion
    """
    et = split_and_normalize(raw_eligibility_text)
    atoms, prov = extract_with_rules(et)

    return {
        "schema": json_schema_for_trial_atoms(),
        "context": {
            "normalized_text": et.normalized,
            "inclusion_items": et.inclusion_items,
            "exclusion_items": et.exclusion_items,
            "rule_extraction_atoms": atoms.to_dict(),
            "provenance": prov.data,
            "notes": [
                "Fields not present remain null/empty; LLM may fill strictly from provided text.",
                "Use inclusion vs exclusion to set polarity (required vs excluded).",
                "Numeric units are canonicalized where possible (e.g., ANC per uL)."
            ]
        }
    }
