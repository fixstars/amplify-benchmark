from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class SolverSolutionMock:
    energy: float = 0.0
    frequency: int = 1
    is_feasible: bool = True
    values: list[int] = field(default_factory=list)


@pytest.fixture
def solver_solution_mock():
    return SolverSolutionMock
