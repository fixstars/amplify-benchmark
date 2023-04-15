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
