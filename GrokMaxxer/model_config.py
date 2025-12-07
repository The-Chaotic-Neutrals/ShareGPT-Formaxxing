import json
import os
import requests
from datetime import datetime

APP_TITLE = "SynthMaxxer: 'Gork Edition'"

DEFAULT_MODEL = "grok-4.1-fast-non-reasoning"
FALLBACK_MODELS = [
    "grok-4-fast-non-reasoning",
]

BASE_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(BASE_DIR, "grok_tool_config.json")
ICON_FILE = os.path.join(BASE_DIR, "icon.ico")

ALLOWED_ROLES = {"system", "human", "gpt"}

SCHEMA_DESCRIPTION = (
    "Each dataset entry is a JSON object:\n"
    "{\n"
    "  \"conversations\": [\n"
    "    {\"from\": \"system\" | \"human\" | \"gpt\", \"value\": \"...\"},\n"
    "    ...\n"
    "  ]\n"
    "}\n"
    "Rules:\n"
    "- Optional leading system message at index 0.\n"
    "- After the optional system message, roles alternate strictly:\n"
    "    human, gpt, human, gpt, ...\n"
    "- The final message MUST NOT be from 'human'.\n"
    "- 'value' is always a string.\n"
)

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

def load_models(api_key, preferred_model=None):
    models = []

    if api_key:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            url = "https://api.x.ai/v1/models"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.ok:
                data = resp.json()
                items = data.get("data") or data.get("models") or data
                if isinstance(items, dict):
                    items = items.get("data") or items.get("models") or []
                if isinstance(items, list):
                    for m in items:
                        mid = None
                        if isinstance(m, dict):
                            mid = m.get("id") or m.get("name")
                        if not mid or not isinstance(mid, str):
                            continue
                        if "image" in mid.lower():
                            continue
                        models.append(mid)
        except Exception:
            models = []

        if not models:
            try:
                from xai_sdk import Client
                client = Client(api_key=api_key)
                model_ids = []

                resp = client.models.list()
                items = getattr(resp, "data", None) or getattr(resp, "models", None) or resp
                if isinstance(items, dict):
                    items = getattr(items, "data", None) or getattr(items, "models", None) or items
                if isinstance(items, list):
                    for m in items:
                        mid = getattr(m, "id", None) or getattr(m, "name", None)
                        if not mid and isinstance(m, dict):
                            mid = m.get("id") or m.get("name")
                        if mid and isinstance(mid, str) and "image" not in mid.lower():
                            model_ids.append(mid)

                if model_ids:
                    models = model_ids
            except Exception:
                pass

    if not models:
        models = FALLBACK_MODELS

    seen = set()
    clean = []
    for m in models:
        if m not in seen:
            seen.add(m)
            clean.append(m)

    if not clean:
        clean = [DEFAULT_MODEL]

    if preferred_model and preferred_model in clean:
        selected = preferred_model
    elif DEFAULT_MODEL in clean:
        selected = DEFAULT_MODEL
    else:
        selected = clean[0]

