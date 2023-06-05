from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolverSolutionSimulator:
    energy: float = 0.0
    frequency: int = 1
    is_feasible: bool = True
    values: list[int] = field(default_factory=list)


@dataclass
class SolutionMock:
    energy: float = 0.0
    frequency: int = 1
    values: list[int] = field(default_factory=list)


class TimingMock:
    def __call__(self) -> float:
        return 1000.0


@dataclass
class ClientResult:
    solutions: list[SolutionMock] = field(default_factory=list)
    timing: TimingMock = field(default_factory=TimingMock)
