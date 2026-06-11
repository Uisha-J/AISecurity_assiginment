"""Compatibility package for the repository-root source layout.

The project modules live at the repository root (defense/, training/, etc.),
while the CLI and tests import them as ``voice_defense.*``. Extending this
package path lets those imports resolve without moving the existing folders.
"""

from __future__ import annotations

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
