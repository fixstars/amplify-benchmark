from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class GurobiClientConfig(ClientConfig):
    _timeout_like_name: str = "time_limit"
