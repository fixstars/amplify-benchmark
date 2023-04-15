from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class FujitsuDA4SolverClientConfig(ClientConfig):
    _timeout_like_name: str = "time_limit_sec"
    _extract_keys: tuple = (
        "set_inequalities",
        "set_one_way_one_hot_groups",
        "set_two_way_one_hot_groups",
        "set_penalty_binary_polynomial",
    )  # settings の内でログに含めるべき属性のリスト
