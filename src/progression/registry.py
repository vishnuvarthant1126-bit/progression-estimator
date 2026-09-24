"""A small file-based model registry.

registry/registry.json holds every trained version with its metrics, the data
fingerprint it was trained on, and the promotion decision. Versions are never
overwritten; the champion pointer moves only when the gate approves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "registry" / "registry.json"


@dataclass
class Registry:
    versions: list[dict] = field(default_factory=list)
    champion: str | None = None

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Registry":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(versions=data["versions"], champion=data["champion"])

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"champion": self.champion, "versions": self.versions}, indent=2) + "\n")

    def next_version(self) -> str:
        return f"v{len(self.versions) + 1}"

    def get(self, version: str) -> dict:
        for v in self.versions:
            if v["version"] == version:
                return v
        raise KeyError(version)

    @property
    def champion_entry(self) -> dict | None:
        return self.get(self.champion) if self.champion else None

    def register(self, entry: dict, promote: bool) -> dict:
        entry = dict(entry, version=self.next_version())
        self.versions.append(entry)
        if promote:
            self.champion = entry["version"]
        return entry
