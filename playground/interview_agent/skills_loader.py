from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

SKILLS_DIR = Path(__file__).parent / "skills"


class SkillLoader:
    """Loads behavior skills from <skills_dir>/<name>/SKILL.md (frontmatter stripped).
    Edit the markdown to retune agent behavior without touching code."""

    def __init__(self, skills_dir: str | Path = SKILLS_DIR):
        self.dir = Path(skills_dir)
        self._cache: Dict[str, str] = {}

    def load(self, name: str) -> str:
        if name not in self._cache:
            path = self.dir / name / "SKILL.md"
            if not path.exists():
                raise FileNotFoundError(f"skill not found: {path}")
            text = path.read_text(encoding="utf-8")
            self._cache[name] = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S).strip()
        return self._cache[name]
