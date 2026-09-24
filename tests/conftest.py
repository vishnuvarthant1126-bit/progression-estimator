import json
from pathlib import Path

import pytest

from progression import data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ds():
    return data.load()


@pytest.fixture(scope="session")
def payload():
    return json.loads((ROOT / "web" / "model.json").read_text())


@pytest.fixture(scope="session")
def registry_json():
    return json.loads((ROOT / "registry" / "registry.json").read_text())
