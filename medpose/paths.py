"""Project locations. Environment variables let tests use a temporary Core checkout."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE_ROOT = Path(os.environ.get("MEDPOSE_CORE_ROOT", ROOT / "opencap-core"))
MONO_ROOT = Path(os.environ.get("MEDPOSE_MONO_ROOT", ROOT / "opencap-monocular"))
# Ignored by git and denied to Claude's file tools; see docs/DATA_POLICY.md.
PATIENT_ROOT = Path(os.environ.get("MEDPOSE_PATIENT_ROOT", ROOT / "patient_data"))
NORMALIZED_ROOT = PATIENT_ROOT / "normalized"
