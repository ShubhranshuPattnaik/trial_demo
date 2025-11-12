import re
from typing import List, Tuple
from .schemas import EligibilityText

ABBREV = {
    r"\byrs?\b": "years",
    r"\bYO\b": "years old",
    r"\bpts\b": "patients",
    r"\bECOG\b": "ECOG",
    r"\bULN\b": "ULN",
    r"\bHg\b": "hemoglobin",
    r"\bHgb\b": "hemoglobin",
    r"\bPLT\b": "platelets",
    # only expand standalone ANC, not when it's already inside parentheses
    r"(?<!\()(?<!\w)\bANC\b(?!\w)(?!\))": "absolute neutrophil count",
}

def _expand_abbrev(text: str) -> str:
    t = text
    for pat, rep in ABBREV.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

def normalize_text(raw: str) -> str:
    t = raw.replace("–","-").replace("—","-").replace("µ","u")
    t = re.sub(r"\r\n|\r", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = _expand_abbrev(t)
    return t.strip()

def _split_on_headers(text: str) -> Tuple[str, str]:
    low = text.lower()
    i_inc = low.find("inclusion")
    i_exc = low.find("exclusion")
    if i_inc == -1 and i_exc == -1:
        return text, ""
    if i_inc != -1 and i_exc != -1:
        if i_inc < i_exc:
            return text[i_inc:i_exc], text[i_exc:]
        else:
            return "", text[i_exc:]
    if i_inc != -1:
        return text[i_inc:], ""
    return "", text[i_exc:]

def _bulletize(block: str) -> List[str]:
    if not block:
        return []
    lines = [l.strip("-*• ").strip() for l in block.split("\n") if l.strip()]
    # Drop lines that are just section headers
    header_re = re.compile(r"^\s*(inclusion|exclusion)\s+criteria:?\s*$", re.IGNORECASE)
    lines = [l for l in lines if not header_re.match(l)]
    return [l for l in lines if len(l) > 3]

def split_and_normalize(raw: str) -> EligibilityText:
    norm = normalize_text(raw)
    inc_block, exc_block = _split_on_headers(norm)
    inc_items = _bulletize(inc_block)
    exc_items = _bulletize(exc_block)
    return EligibilityText(raw=raw, normalized=norm, inclusion_items=inc_items, exclusion_items=exc_items)
