"""
Central configuration loader.
Loads configs/default.yaml and allows dot-notation access.
"""
import yaml
from pathlib import Path
from typing import Any


class Config:
    """
    Recursive dot-notation config wrapper around a YAML dict.

    Usage
    -----
    cfg = Config.load("configs/default.yaml")
    print(cfg.dae.roberta.learning_rate)   # 2e-5
    print(cfg.prta.ddqn.batch_size)        # 64
    """

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    # Factory
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from YAML file. Defaults to configs/default.yaml."""
        if path is None:
            # Walk up until we find the configs directory
            here = Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "configs" / "default.yaml"
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(
                    "Could not locate configs/default.yaml. "
                    "Pass the path explicitly."
                )
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    
    # Helpers
    def to_dict(self) -> dict:
        """Recursively convert back to a plain dict."""
        result = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({list(self.__dict__.keys())})"
