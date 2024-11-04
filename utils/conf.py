from typing import Dict, Any
import os
import yaml
from easydict import EasyDict

from utils.log import INFO, ERROR

__all__ = ["read_conf"]

def make_easy(mapping: Dict[str, Any]) -> EasyDict:
    """
    Utility function for recursively creating an ``EasyDict`` out of a ``Dict``.

    :param mapping The ``dict`` object to convert.
    :return The converted ``dict`` object as an ``EasyDict`` instance.
    """
    
    for key, value in mapping.items():
        if isinstance(value, dict):
            value = make_easy(value)
        mapping[key] = value
    return EasyDict(mapping)

def read_conf(path: str) -> EasyDict:
    """
    Read a ``.yaml`` configuration file.

    :param path The path of the configuration file to read.
    :return The read configuration as an ``EasyDict``.
    """
    if not os.path.exists(path):
        ERROR(f"The specified configuration file doesn't exist: {path}")
        return None
    
    with open(path, 'r') as infd:
        conf = yaml.safe_load(infd)
    conf = make_easy(conf)
    return conf