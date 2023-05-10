import pytest
from click.testing import CliRunner


@pytest.fixture(scope="module")
def runner():
    return CliRunner()
