from pathlib import Path

import pytest


@pytest.fixture
def data() -> Path:
    DATA_DIR = Path(__file__).parent / ".." / ".." / "src" / "benchmark" / "problem" / "data"
    return DATA_DIR
