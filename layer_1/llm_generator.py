"""
layer_1/llm_generator.py
-------------------------
LLM-powered scenario planner for Layer 1.

Given a user's natural language prompt (e.g. "accelerate on highway, hit a pothole, then crash"),
this module calls the Gemini API and converts the response into a list of
structured segment dicts that data_generator.generate_from_segments() can consume.

If the API key is missing or the call fails, a keyword-matching fallback is used instead.

Supported events (must match EVENT_REGISTRY in data_generator.py):
    Normal:     normal, highway, city, accel, rain
    Near-Crash: pothole, swerve, brake, gravel
    Crash:      crash, highside, lowside, front_collision, rear_collision
"""

import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

# ─── Gemini SDK (optional) ────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────
VALID_EVENTS = {
    # Normal (label 0)
    "normal", "highway", "city", "accel", "rain",
    # Near-Crash (label 1)
    "pothole", "swerve", "brake", "gravel",
    # Crash (label 2)
    "crash", "highside", "lowside", "front_collision", "rear_collision",
}

# Keyword → event name for the fallback parser
_KEYWORD_MAP = {
    "highway":       "highway",
    "highway cruise":"highway",
    "city":          "city",
    "urban":         "city",
    "rain":          "rain",
    "wet":           "rain",
    "accelerat":     "accel",
    "speed up":      "accel",
    "pothole":       "pothole",
    "bump":          "pothole",
    "brake":         "brake",
    "braking":       "brake",
    "stop":          "brake",
    "swerve":        "swerve",
    "dodge":         "swerve",
    "gravel":        "gravel",
    "slip":          "gravel",
    "traction":      "gravel",
    "crash":         "crash",
    "collision":     "front_collision",
    "front":         "front_collision",
    "impact":        "front_collision",
    "rear":          "rear_collision",
    "hit from behind": "rear_collision",
    "highside":      "highside",
    "high-side":     "highside",
    "lowside":       "lowside",
    "low-side":      "lowside",
    "slide":         "lowside",
    "fall":          "lowside",
}

# ─── System Prompt sent to Gemini ─────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a motorcycle riding scenario planner for a smart airbag helmet simulation system.

Given a natural language description of a ride scenario, your task is to break it down
into a sequence of riding segments. Each segment is one of the following named events:

NORMAL (label 0):  normal, highway, city, accel, rain
NEAR-CRASH (label 1): pothole, swerve, brake, gravel
CRASH (label 2): crash, highside, lowside, front_collision, rear_collision

You MUST respond with ONLY a valid JSON array. Each element must have exactly two keys:
  "event"       : one of the valid event names above (string)
  "duration_ms" : duration of this segment in milliseconds (integer, min 100, max 5000)

Rules:
- Always start with a normal/highway/city segment.
- If the scenario ends with a crash, the crash segment should be the last one.
- Total duration should be realistic (1000ms to 15000ms total).
- Do NOT include any explanation, markdown, or text outside the JSON array.

Example output:
[
  {"event": "highway",  "duration_ms": 1500},
  {"event": "pothole",  "duration_ms": 400},
  {"event": "normal",   "duration_ms": 800},
  {"event": "crash",    "duration_ms": 500}
]
"""


def _call_gemini(prompt: str, api_key: str) -> list[dict]:
    """Call the Gemini API and parse the returned JSON segment list."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-3.6-flash",
        system_instruction=_SYSTEM_PROMPT,
    )
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Strip markdown code fences if present
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r"```\s*$", "", raw_text, flags=re.MULTILINE).strip()

    segments = json.loads(raw_text)
    return _validate_segments(segments)


def _keyword_fallback(prompt: str) -> list[dict]:
    """
    Simple keyword matching fallback when Gemini is unavailable.
    Scans the prompt for known keywords and builds a rough segment list.
    """
    print("[LLM FALLBACK] Gemini unavailable — using keyword matcher.")
    prompt_lower = prompt.lower()

    found_events = []
    for keyword, event in _KEYWORD_MAP.items():
        if keyword in prompt_lower:
            found_events.append(event)

    # Deduplicate while preserving order
    seen = set()
    unique_events = []
    for e in found_events:
        if e not in seen:
            seen.add(e)
            unique_events.append(e)

    if not unique_events:
        print("[LLM FALLBACK] No keywords matched — defaulting to normal highway.")
        unique_events = ["highway"]

    # Ensure starts with a normal event
    normal_events = {"normal", "highway", "city", "accel", "rain"}
    if unique_events[0] not in normal_events:
        unique_events.insert(0, "highway")

    # Build segment list with realistic cinematic durations
    # Normal ride: 8-12 seconds so you can SEE the bike riding
    # Near-crash:  3-4 seconds — enough to feel the tension
    # Crash:       3 seconds   — clear impact and deploy
    segments = []
    for event in unique_events:
        if event in {"crash", "highside", "lowside", "front_collision", "rear_collision"}:
            dur = 3000
        elif event in {"pothole", "swerve", "brake", "gravel"}:
            dur = 3500
        else:
            dur = 9000    # 9 seconds of normal riding
        segments.append({"event": event, "duration_ms": dur})

    return segments


def _validate_segments(segments: list) -> list[dict]:
    """Validate and sanitize a list of segment dicts."""
    clean = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        event = str(seg.get("event", "normal")).lower().replace(" ", "_").replace("-", "_")
        if event not in VALID_EVENTS:
            print(f"[WARN] Unknown event '{event}' from LLM — skipping.")
            continue
        duration_ms = max(500, min(15000, int(seg.get("duration_ms", 9000))))
        clean.append({"event": event, "duration_ms": duration_ms})
    return clean if clean else [{"event": "highway", "duration_ms": 1000}]


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_segments_from_prompt(prompt: str) -> list[dict]:
    """
    Convert a natural language prompt into a list of riding segments.

    Tries the Gemini API first (reads GEMINI_API_KEY from environment).
    Falls back to keyword matching if Gemini is unavailable.

    Parameters
    ----------
    prompt : str
        Free-form natural language description of the ride scenario.

    Returns
    -------
    list of dict  [{"event": str, "duration_ms": int}, ...]
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        # Check workspace root .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("GEMINI_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if _GENAI_AVAILABLE and api_key:
        try:
            print(f"[LLM] Sending prompt to Gemini: \"{prompt}\"")
            segments = _call_gemini(prompt, api_key)
            print(f"[LLM] Gemini returned {len(segments)} segments.")
            return segments
        except Exception as e:
            print(f"[LLM ERROR] Gemini call failed: {e}. Falling back to keyword matcher.")

    return _keyword_fallback(prompt)
