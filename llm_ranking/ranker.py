"""
llm_ranking/ranker.py

Consumes the `integration_data` produced by /search/hybrid_v2 and returns,
for each trial, a verdict (include/unsure/exclude), an eligibility_score [0,1],
and short notes + unmet_criteria (if any).

Provider-agnostic with simple adapters:
- OPENAI (requires OPENAI_API_KEY)
- OLLAMA (local http://localhost:11434, model e.g. "llama3")
- LOCAL_HEURISTIC (no LLM; deterministic fallback)

Env (sane defaults):
  LLM_PROVIDER=OPENAI|OLLAMA|LOCAL_HEURISTIC (default: LOCAL_HEURISTIC)
  LLM_MODEL=gpt-4o-mini (OPENAI) | llama3 (OLLAMA)
  LLM_MAX_TOKENS=600
  LLM_TEMPERATURE=0.1
"""

import json
import os
import re
import time
from typing import Any, Dict, List

import requests

from config.logging_config import quick_setup

logger = quick_setup("llm_ranking")

# -------------------- config --------------------
PROVIDER = os.getenv("LLM_PROVIDER", "LOCAL_HEURISTIC").upper()
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini" if PROVIDER == "OPENAI" else "llama3")
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Small, in-memory cache to avoid recomputing identical requests
_CACHE: Dict[str, Dict[str, Any]] = {}

# -------------------- prompt --------------------
SYSTEM_PROMPT = (
    "You are a clinical-trial eligibility assistant. "
    "Given a patient note and compact trial atoms/evidence, return JSON ONLY. "
    "Judge patient eligibility for EACH trial id with fields: "
    "`trial_id`, `verdict` (include|unsure|exclude), `eligibility_score` (0-1), "
    "`unmet_criteria` (array), and `notes` (short rationale). Be concise and calibrated."
)

def _truncate_list(items: List[str], limit: int) -> List[str]:
    if not items:
        return []
    return items[:limit]

def build_prompt(integration_data: Dict[str, Any]) -> str:
    """
    Creates a compact, deterministic prompt so outputs are stable & cheap.
    """
    patient = integration_data.get("patient_text", "").strip()
    trials = integration_data.get("trials", []) or []

    # Keep prompt small: cap evidence lines per trial
    lines = []
    lines.append("PATIENT_NOTE:")
    lines.append(patient)

    lines.append("\nTRIALS:")
    for t in trials:
        tid = t.get("trial_id") or t.get("nct_id") or "UNKNOWN"
        atoms = t.get("atoms_subset", {}) or {}
        ev = t.get("evidence", {}) or {}
        inc = _truncate_list(ev.get("inclusion_items", []), 6)
        exc = _truncate_list(ev.get("exclusion_items", []), 6)

        lines.append(f"\nTRIAL_ID: {tid}")
        lines.append(f"TITLE: {t.get('title','')}")
        lines.append("ATOMS_SUBSET:")
        #keep atoms compact + deterministic ordering
        for k in sorted(atoms.keys()):
            lines.append(f"  - {k}: {atoms[k]}")

        if inc:
            lines.append("INCLUSION_SAMPLES:")
            for it in inc:
                lines.append(f"  - {it}")
        if exc:
            lines.append("EXCLUSION_SAMPLES:")
            for it in exc:
                lines.append(f"  - {it}")

    lines.append(
        "\nRETURN JSON EXACTLY IN THIS SHAPE:\n"
        "{\n"
        '  "patient_text": "<echo short gist>",\n'
        '  "trials": [\n'
        "    {\n"
        '      "trial_id": "<id>",\n'
        '      "verdict": "include|unsure|exclude",\n'
        '      "eligibility_score": 0.0,\n'
        '      "unmet_criteria": [],\n'
        '      "notes": "short rationale"\n'
        "    }\n"
        "  ]\n"
        "}"
    )
    return "\n".join(lines)

# -------------------- providers --------------------
def _strip_code_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.MULTILINE)

def _normalize_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure required keys exist; clamp eligibility_score; normalize verdict.
    """
    trials = obj.get("trials", []) or []
    norm = []
    for t in trials:
        tid = t.get("trial_id", "UNKNOWN")
        verdict = (t.get("verdict") or "unsure").lower()
        if verdict not in {"include", "unsure", "exclude"}:
            verdict = "unsure"
        score = float(t.get("eligibility_score", 0.5))
        score = max(0.0, min(1.0, score))
        unmet = t.get("unmet_criteria") or []
        if not isinstance(unmet, list):
            unmet = [str(unmet)]
        notes = (t.get("notes") or "").strip()
        norm.append({
            "trial_id": tid,
            "verdict": verdict,
            "eligibility_score": score,
            "unmet_criteria": unmet,
            "notes": notes[:300]
        })
    return {"patient_text": obj.get("patient_text", ""), "trials": norm}

def _call_openai(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    import openai  # type: ignore
    openai.api_key = api_key

    # Chat Completions
    resp = openai.ChatCompletion.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp["choices"][0]["message"]["content"]
    text = _strip_code_fences(content)
    return _normalize_output(json.loads(text))

def _call_ollama(prompt: str) -> Dict[str, Any]:
    """
    Local LLM via Ollama (http://localhost:11434).
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    # Streaming or single — handle both
    if isinstance(r.json(), dict) and "message" in r.json():
        content = r.json()["message"]["content"]
    else:
        # If streaming chunked, concat all `message.content`
        content = ""
        for line in r.iter_lines():
            try:
                j = json.loads(line)
                content += j.get("message", {}).get("content", "")
            except Exception:
                pass
    text = _strip_code_fences(content)
    return _normalize_output(json.loads(text))

def _call_local_heuristic(integration_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    No-LLM fallback: fast, deterministic heuristic for demos.
    Very conservative: prefers 'unsure' unless strong atom match/mismatch.
    """
    patient = integration_data.get("patient_text", "")
    trials = integration_data.get("trials", []) or []

    out = {"patient_text": patient[:160], "trials": []}
    p = patient.lower()

    for t in trials:
        tid = t.get("trial_id", "UNKNOWN")
        atoms = t.get("atoms_subset", {}) or {}
        notes = []
        score = 0.5  # start neutral
        verdict = "unsure"

        # quick constraints
        sex = (atoms.get("sex") or "").lower()
        if sex in {"male", "female"} and sex not in {"", "all", "unknown"}:
            if (" female" in p and sex == "female") or (" male" in p and sex == "male"):
                score += 0.1
            else:
                notes.append(f"sex={sex} constraint may not match patient")
                score -= 0.2

        perf = atoms.get("performance", {})
        if isinstance(perf, dict) and "ecog_max" in perf:
            # assume typical ECOG 0-2 unless text says otherwise
            score += 0.05

        # semantic nudge based on disease words overlap
        title = (t.get("title") or "").lower()
        if any(w in p for w in title.split()[:5]):
            score += 0.05

        # If explicit cancer-type mismatch words in atoms — nudge down
        if "melanoma" in p and "breast" in title:
            score -= 0.4
            notes.append("cancer type mismatch (melanoma vs breast)")

        score = max(0.0, min(1.0, score))
        if score >= 0.7:
            verdict = "include"
        elif score <= 0.3:
            verdict = "exclude"

        out["trials"].append({
            "trial_id": tid,
            "verdict": verdict,
            "eligibility_score": float(round(score, 3)),
            "unmet_criteria": [] if verdict == "include" else notes,
            "notes": "heuristic fallback (no LLM)"
        })
    return out

# -------------------- public API --------------------
def llm_rank_trials(integration_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry: given `integration_data` from /search/hybrid_v2,
    return normalized JSON verdicts.
    """
    # cache key — stable hash from data
    key = json.dumps(integration_data, sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]

    prompt = build_prompt(integration_data)

    t0 = time.time()
    try:
        if PROVIDER == "OPENAI":
            result = _call_openai(prompt)
        elif PROVIDER == "OLLAMA":
            result = _call_ollama(prompt)
        else:
            result = _call_local_heuristic(integration_data)
    except Exception as e:
        logger.warning(f"LLM provider failed ({PROVIDER}): {e}. Falling back to heuristic.")
        result = _call_local_heuristic(integration_data)

    dt = (time.time() - t0)
    logger.info(f"llm_rank_trials provider={PROVIDER} model={MODEL} time_sec={dt:.2f}")

    _CACHE[key] = result
    return result
