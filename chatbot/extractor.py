import json
import re
import requests
from typing import Dict, List, Any

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


LOCKED_ALWAYS = {"NAME", "host name", "last review"}


NUMERIC_FIELDS = [
    "Construction year",
    "service fee",
    "minimum nights",
    "number of reviews",
    "reviews per month",
    "review rate number",
    "calculated host listings count",
    "availability 365",
]

BOOLEAN_FIELDS = {"instant_bookable", "host_identity_verified"}


def _ollama(prompt: str, timeout: int = 180) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found:\n{text}")
    return json.loads(match.group(0))


def normalize_bool(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "Unknown"
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return "True"
    if s in {"false", "f", "no", "n", "0"}:
        return "False"
    return "Unknown"


def parse_number(text: str, pattern: str):
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def hard_parse_numerics(user_text: str) -> Dict[str, float]:
    t = user_text.lower()

    out = {}

    val = parse_number(t, r"(?:construction year|year built)\s*[:=]?\s*(\d{4})")
    if val:
        out["Construction year"] = float(val)

    val = parse_number(t, r"(?:service fee)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["service fee"] = float(val)

    val = parse_number(t, r"(?:minimum nights|min nights)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["minimum nights"] = float(val)

    val = parse_number(t, r"(?:number of reviews|reviews)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["number of reviews"] = float(val)

    val = parse_number(t, r"(?:reviews per month)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["reviews per month"] = float(val)

    val = parse_number(t, r"(?:review rate number|rating|review score)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["review rate number"] = float(val)

    val = parse_number(t, r"(?:calculated host listings count|host listings)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["calculated host listings count"] = float(val)

    val = parse_number(t, r"(?:availability\s*365|availability)\s*[:=]?\s*(\d+(\.\d+)?)")
    if val is not None:
        out["availability 365"] = float(val)

    return out


def llm_extract_categoricals(
    user_text: str,
    allowed_categoricals: List[str],
) -> Dict[str, Any]:
    prompt = f"""
You extract ONLY categorical Airbnb features from user text.

Output ONLY one JSON object.
Allowed keys:
{allowed_categoricals}

Rules:
- Only allowed keys.
- Omit unknown keys.
- Do NOT include NAME, host name, last review.
- For boolean fields: instant_bookable and host_identity_verified return true/false.

Mapping:
- "entire apartment" => room type = "Entire home/apt"
- "private room" => room type = "Private room"
- "shared room" => room type = "Shared room"
- Assume country is United States if not mentioned.

User message:
{user_text}

JSON:
""".strip()

    raw = _ollama(prompt)
    extracted = _extract_json(raw)

    clean = {}
    for k, v in extracted.items():
        if k not in allowed_categoricals:
            continue
        if k in LOCKED_ALWAYS:
            continue
        clean[k] = v

    return clean


def extract_features(
    user_text: str,
    categorical_features: List[str],
    numerical_features: List[str],
) -> Dict:
    extracted = {}

    extracted.update(hard_parse_numerics(user_text))
    extracted.update(llm_extract_categoricals(user_text, categorical_features))

    extracted.pop("NAME", None)
    extracted.pop("host name", None)
    extracted.pop("last review", None)

    return extracted


def apply_defaults(
    features: Dict,
    categorical_features: List[str],
    numerical_features: List[str],
) -> Dict:
    smart_numeric_defaults = {
        "minimum nights": 2.0,
        "availability 365": 180.0,
        "number of reviews": 20.0,
        "reviews per month": 2.0,
        "review rate number": 4.2,
        "service fee": 20.0,
        "calculated host listings count": 1.0,
        "Construction year": 2015.0,
        "lat": 0.0,
        "long": 0.0,
        "id": 0.0,
        "host id": 0.0,
    }

    final = {}

    for c in categorical_features:
        if c in LOCKED_ALWAYS:
            final[c] = "Unknown"
            continue

        val = features.get(c)

        if c in BOOLEAN_FIELDS:
            final[c] = normalize_bool(val)
        else:
            final[c] = "Unknown" if val is None or str(val).strip() == "" else str(val)

    for n in numerical_features:
        raw = features.get(n, smart_numeric_defaults.get(n, 0.0))
        try:
            final[n] = float(raw)
        except Exception:
            final[n] = float(smart_numeric_defaults.get(n, 0.0))

    if "country" in final and final["country"] in {"Unknown", ""}:
        final["country"] = "United States"
    if "country code" in final and final["country code"] in {"Unknown", ""}:
        final["country code"] = "US"

    return final


