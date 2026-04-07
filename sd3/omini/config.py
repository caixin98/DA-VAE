"""Shared configuration utilities for DA-VAE training pipelines."""

import os
from typing import Dict


def get_config() -> Dict:
    """Load YAML configuration from the path specified by OMINI_CONFIG env var."""
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"

    try:
        from ruamel.yaml import YAML
        yaml_parser = YAML(typ="safe")
        yaml_parser.preserve_quotes = True
        with open(config_path, "r") as f:
            config = yaml_parser.load(f)
    except Exception:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    return config


class SimpleConfig:
    """Adapter that converts a dict into an attribute-accessible object."""

    def __init__(self, config_dict):
        for key, value in (config_dict or {}).items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)
