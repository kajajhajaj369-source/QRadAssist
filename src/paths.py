"""Repository path helpers.

Assumption: this file lives at <project_root>/src/paths.py.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
DEFAULT_DISEASES_FILE = CONFIG_DIR / "diseases.json"
