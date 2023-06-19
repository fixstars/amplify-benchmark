from pathlib import Path

import pytest
from amplify import BinarySymbolGenerator
from amplify.constraint import equal_to


@pytest.fixture
def simple_problem():
    gen = BinarySymbolGenerator()
    q = gen.array(4)
    binary_poly = -q[0] * q[1] + 99.0
    binary_constraints = equal_to(q[2], 1) + equal_to(q[3], 1)
    model = binary_poly + binary_constraints
    return model


@pytest.fixture
def data() -> Path:
    DATA_DIR = Path(__file__).parent / ".." / "src" / "benchmark" / "problem" / "data"
    return DATA_DIR


@pytest.fixture
def cleanup():
    def _cleanup(filepath: str):
        print(f"clean up file: {filepath}")
        Path(filepath).unlink(missing_ok=True)

    return _cleanup


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
