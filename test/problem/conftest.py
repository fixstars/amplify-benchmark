from pathlib import Path

import pytest


@pytest.fixture
def data() -> Path:
    DATA_DIR = Path(__file__).parent / ".." / ".." / "src" / "benchmark" / "problem" / "data"
    return DATA_DIR


@pytest.fixture
def cleanup():
    def _cleanup(filepath: str):
        print(f"clean up file: {filepath}")
        Path(filepath).unlink(missing_ok=True)

    return _cleanup
