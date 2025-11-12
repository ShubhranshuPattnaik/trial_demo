# eligibility_atoms/rules.py
import re
from typing import Optional, List
from .schemas import TrialAtoms, SexAllowed, Provenance, EligibilityText

def _to_int(s: str):
    try:
        return int(float(s))
    except Exception:
        return None

def _to_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def _weeks_to_days(n: int) -> int:
    return int(n * 7)

def _per_uL_from_10e9_L(val: float) -> int:
    # 1 x 10^9 / L == 1000 / uL
    return int(round(val * 1000))

def _find(pattern: str, text: str, flags=re.IGNORECASE):
    return re.search(pattern, text, flags)

# Map common drug names to therapy classes
DRUG2CLASS = {
    "palbociclib": "CDK4/6 inhibitor",
    "ribociclib": "CDK4/6 inhibitor",
    "abemaciclib": "CDK4/6 inhibitor",
    "pembrolizumab": "PD-1 inhibitor",
    "nivolumab": "PD-1 inhibitor",
    "atezolizumab": "PD-L1 inhibitor",
    "durvalumab": "PD-L1 inhibitor",
}

# Robust biomarker patterns
# Notes:
# - No trailing \b after '+' or '-' (they are non-word chars)
# - Accept unicode minus: -, −, –
# - Include English forms like "HER2 negative" / "estrogen receptor positive"
BIOMARKER_SYNONYMS = {
    r"(?<!\w)ER\s*\+": "ER+",
    r"\bestrogen\s+receptor(?:\s*[- ]*positive|\s*\+)\b": "ER+",
    r"(?<!\w)HER2\s*[-−–]": "HER2-",
    r"\bHER2\s*negative\b": "HER2-",
    r"(?<!\w)HER2\s*\+": "HER2+",
    r"\bHER2\s*positive\b": "HER2+",
    r"\bPD-?L1\s*>=?\s*1%\b": "PD-L1>=1%",
    r"\bBRAF\s*V600E\b": "BRAF V600E",
}

EXCLUSION_PHRASES = [
    "active autoimmune disease",
    "pregnancy",
    "uncontrolled brain metastases",
    "active infection",
]

# -------- Sex detection helper (boundary-safe, looks at bullets first) --------
def _detect_sex_from_lines(lines: List[str]) -> Optional[SexAllowed]:
    """
    Infer allowed sex from typical phrases across bullets.
    Uses word boundaries so 'male' won't match inside 'female'.
    """
    if not lines:
        return None
    txt = "\n".join(lines)

    # explicit “both” style phrases override
    if re.search(r"\b(both\s+sexes|all\s+sexes|male\s+or\s+female|men\s+and\s+women)\b", txt, re.I):
        return SexAllowed.all

    has_female = re.search(r"\bfemales?\b|\bwomen\b", txt, re.I) is not None
    has_male   = re.search(r"\bmales?\b|\bmen\b", txt, re.I) is not None

    if has_female and not has_male:
        return SexAllowed.female
    if has_male and not has_female:
        return SexAllowed.male
    if has_female and has_male:
        return SexAllowed.all
    return None

def extract_with_rules(et: EligibilityText):
    atoms = TrialAtoms()
    prov = Provenance()
    text = et.normalized

    # ----- Age -----
    m = _find(r"(?:age|aged)\s*(\d{1,3})\s*(?:to|-|–|−)\s*(\d{1,3})\s*years", text)
    if m:
        atoms.min_age_years = _to_int(m.group(1))
        atoms.max_age_years = _to_int(m.group(2))
        prov.add("min_age_years", atoms.min_age_years, m.group(0), "both")
        prov.add("max_age_years", atoms.max_age_years, m.group(0), "both")
    else:
        m = _find(r"(?:age|aged)\s*(?:≥|>=|at least)\s*(\d{1,3})\s*years", text)
        if m:
            atoms.min_age_years = _to_int(m.group(1))
            prov.add("min_age_years", atoms.min_age_years, m.group(0), "both")
        m = _find(r"(?:age|aged)\s*(?:≤|<=|at most|up to)\s*(\d{1,3})\s*years", text)
        if m:
            atoms.max_age_years = _to_int(m.group(1))
            prov.add("max_age_years", atoms.max_age_years, m.group(0), "both")

    # ----- Sex (prefer bullets; fallback to free text) -----
    sex_any = _detect_sex_from_lines(et.inclusion_items + et.exclusion_items)
    if sex_any is not None:
        atoms.sex_allowed = sex_any
        prov.add("sex_allowed",
                 sex_any.value if hasattr(sex_any, "value") else str(sex_any),
                 "sex mention in bullets", "both")
    else:
        low_all = text.lower()
        if re.search(r"\bfemale only\b|\bwomen only\b", low_all):
            atoms.sex_allowed = SexAllowed.female; prov.add("sex_allowed", "female", "female only", "both")
        elif re.search(r"\bmale only\b|\bmen only\b", low_all):
            atoms.sex_allowed = SexAllowed.male; prov.add("sex_allowed", "male", "male only", "both")
        elif re.search(r"\b(both|all)\s+sexes\b|male\s+or\s+female|men\s+and\s+women", low_all):
            atoms.sex_allowed = SexAllowed.all; prov.add("sex_allowed", "all", "both sexes", "both")

    # ----- ECOG -----
    m = _find(r"ECOG(?:\s*PS|\s*performance status)?\s*(?:≤|<=)?\s*([0-4])", text)
    if m:
        atoms.ecog_max = _to_int(m.group(1)); prov.add("ecog_max", atoms.ecog_max, m.group(0), "both")
    else:
        m = _find(r"ECOG\s*([0-4])\s*[-−–]\s*([0-4])", text)
        if m:
            atoms.ecog_max = max(_to_int(m.group(1)), _to_int(m.group(2)))
            prov.add("ecog_max", atoms.ecog_max, m.group(0), "both")

    # ----- Line-by-line parse for labs, biomarkers, therapies, windows, exclusions -----
    for line in et.inclusion_items + et.exclusion_items:
        low = line.lower()

        # ANC: >= 1.5 x 10^9/L  OR  >= 1500 /uL
        m = _find(r"(absolute neutrophil count|ANC).*(?:≥|>=)\s*(\d+(?:\.\d+)?)\s*x?\s*10\^?9\s*/\s*L", line)
        if m:
            anc = _per_uL_from_10e9_L(float(m.group(2)))
            atoms.lab_thresholds.anc_per_uL_min = anc
            prov.add("lab_thresholds.anc_per_uL_min", anc, line, "inc")
        m = _find(r"(absolute neutrophil count|ANC).*(?:≥|>=)\s*(\d{3,5})\s*/?\s*u?L", line)
        if m and atoms.lab_thresholds.anc_per_uL_min is None:
            anc = _to_int(m.group(2))
            atoms.lab_thresholds.anc_per_uL_min = anc
            prov.add("lab_thresholds.anc_per_uL_min", anc, line, "inc")

        # Platelets
        m = _find(r"platelet[s]?(?:\s*count)?\s*(?:≥|>=)\s*(\d{3,6})\s*/?\s*u?L", line)
        if m:
            atoms.lab_thresholds.platelets_per_uL_min = _to_int(m.group(1))
            prov.add("lab_thresholds.platelets_per_uL_min", atoms.lab_thresholds.platelets_per_uL_min, line, "inc")

        # Hemoglobin
        m = _find(r"hemoglobin\s*(?:≥|>=)\s*(\d+(?:\.\d+)?)\s*g\s*/\s*dL", line)
        if m:
            atoms.lab_thresholds.hemoglobin_g_dL_min = _to_float(m.group(1))
            prov.add("lab_thresholds.hemoglobin_g_dL_min", atoms.lab_thresholds.hemoglobin_g_dL_min, line, "inc")

        # Creatinine
        m = _find(r"\bcreatinine\b.*(?:≤|<=)\s*(\d+(?:\.\d+)?)\s*mg\s*/\s*dL", line)
        if m:
            atoms.lab_thresholds.creatinine_mg_dL_max = _to_float(m.group(1))
            prov.add("lab_thresholds.creatinine_mg_dL_max", atoms.lab_thresholds.creatinine_mg_dL_max, line, "inc")

        # AST / ALT
        m = _find(r"\bAST\b.*(?:≤|<=)\s*(\d+(?:\.\d+)?)\s*U\s*/\s*L", line)
        if m:
            atoms.lab_thresholds.ast_uL_max = _to_float(m.group(1))
            prov.add("lab_thresholds.ast_uL_max", atoms.lab_thresholds.ast_uL_max, line, "inc")
        m = _find(r"\bALT\b.*(?:≤|<=)\s*(\d+(?:\.\d+)?)\s*U\s*/\s*L", line)
        if m:
            atoms.lab_thresholds.alt_uL_max = _to_float(m.group(1))
            prov.add("lab_thresholds.alt_uL_max", atoms.lab_thresholds.alt_uL_max, line, "inc")

        # Bilirubin
        m = _find(r"\bbilirubin\b.*(?:≤|<=)\s*(\d+(?:\.\d+)?)\s*mg\s*/\s*dL", line)
        if m:
            atoms.lab_thresholds.bilirubin_mg_dL_max = _to_float(m.group(1))
            prov.add("lab_thresholds.bilirubin_mg_dL_max", atoms.lab_thresholds.bilirubin_mg_dL_max, line, "inc")

        # Washout windows
        m = _find(r"(?:at least|≥|>=)\s*(\d+)\s*(?:days|day)\s*(?:washout|since last (?:therapy|treatment))", line)
        if m:
            days = int(m.group(1))
            prev = atoms.windows.get("washout_days") or 0
            atoms.windows["washout_days"] = max(prev, days)
            prov.add("windows.washout_days", atoms.windows["washout_days"], line, "inc")
        m = _find(r"(?:at least|≥|>=)\s*(\d+)\s*weeks\s*(?:washout|since last (?:therapy|treatment))", line)
        if m:
            days = _weeks_to_days(int(m.group(1)))
            prev = atoms.windows.get("washout_days") or 0
            atoms.windows["washout_days"] = max(prev, days)
            prov.add("windows.washout_days", atoms.windows["washout_days"], line, "inc")

        # Biomarkers
        for pat, tag in BIOMARKER_SYNONYMS.items():
            if re.search(pat, line, flags=re.IGNORECASE):
                if line in et.exclusion_items and tag not in atoms.excluded_biomarkers:
                    atoms.excluded_biomarkers.append(tag); prov.add("excluded_biomarkers", tag, line, "exc")
                elif tag not in atoms.required_biomarkers:
                    atoms.required_biomarkers.append(tag); prov.add("required_biomarkers", tag, line, "inc")

        # Therapy rules (allowed / disallowed)
        for drug, klass in DRUG2CLASS.items():
            if drug in low or klass.lower() in low:
                if any(k in low for k in ["no prior", "not allowed", "exclusion", "disallow"]):
                    if klass not in atoms.disallowed_therapies:
                        atoms.disallowed_therapies.append(klass); prov.add("disallowed_therapies", klass, line, "exc")
                else:
                    if klass not in atoms.required_or_allowed_therapies:
                        atoms.required_or_allowed_therapies.append(klass); prov.add("required_or_allowed_therapies", klass, line, "inc")

        # Common exclusions
        for phrase in EXCLUSION_PHRASES:
            if phrase in low:
                if phrase not in atoms.excluded_conditions:
                    atoms.excluded_conditions.append(phrase); prov.add("excluded_conditions", phrase, line, "exc")

    return atoms, prov
