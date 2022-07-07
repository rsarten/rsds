import json
from typing import Any

from attrs import define, field


@define(kw_only=True)
class DirectoryConfig:
    data: str = field(default="./data/")
    env: str = field(default="./venv/")


@define(kw_only=True)
class Config:
    directories: DirectoryConfig = field(default=DirectoryConfig())
    parameters: dict[str, Any] = field(default={})


def create_config(file: str = "./data/config.json"):
    with open(file, "r", encoding="utf-8") as conf:
        data = json.load(conf)
        if not all(b in data.keys() for b in ["directories", "parameters"]):
            raise KeyError("'directories' and 'parameters' must be keys in config")

        config: dict[str, Any] = {}
        config["directories"] = DirectoryConfig(**data["directories"])
        config["parameters"] = data["parameters"]
        return Config(**config)
