"""Load configs: YAML file deep-merged over configs/default.yaml."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs" / "default.yaml"


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, overrides=None):
    with open(DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_PATH.resolve():
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f))
    return deep_merge(cfg, overrides)


def set_dotted(cfg, key, value):
    """set_dotted(cfg, "events.cooldown", 100)"""
    node = cfg
    *parents, last = key.split(".")
    for p in parents:
        node = node[p]
    node[last] = value


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
