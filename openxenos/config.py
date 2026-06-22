"""
Configuration — reads API keys and model settings from environment variables.

Loads .env file if present (for launchd auto-start).

Expected env vars:
    ANTHROPIC_BASE_URL   — Anthropic Messages API endpoint (any compatible provider)
    ANTHROPIC_AUTH_TOKEN — API key
    ANTHROPIC_MODEL      — model name (e.g. claude-sonnet-4-6)
"""

import os
from pathlib import Path

# Load .env from project root if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key not in os.environ:
                    os.environ[key] = val

# ── Anthropic Messages API ──────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ── Deliberation parameters ────────────────────────────────────
PANEL_SIZE = int(os.getenv("OPENXENOS_PANEL_SIZE", "3"))
PHASE1_TEMPERATURE = float(os.getenv("OPENXENOS_PHASE1_TEMP", "0.8"))
PHASE2_TEMPERATURE = float(os.getenv("OPENXENOS_PHASE2_TEMP", "0.5"))
MAX_ROUNDS = int(os.getenv("OPENXENOS_MAX_ROUNDS", "10"))
# ── Server ─────────────────────────────────────────────────────
SERVER_HOST = os.getenv("OPENXENOS_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("OPENXENOS_PORT", "2222"))
