from dataclasses import dataclass

from .base import ClientConfig


@dataclass(frozen=True)
class NECClientConfig(ClientConfig):
    _timeout_like_name: str = "num_sweeps"
    _extract_keys: tuple = (
        "set_andzero",
        "set_maxone",
        "set_minmaxone",
        "set_onehot",
        "set_orone",
    )  # settings の内でログに含めるべき属性のリスト
