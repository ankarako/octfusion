__all__ = ['register_dataset', 'get_dataset']

k_dataset_registry = { }

import utils.log as log

def register_dataset(cls):
    """
    Decorator for registering dataset
    classes in the registry
    """
    k_dataset_registry[cls.__name__] = cls
    return cls

def get_dataset(key: str, **kwargs):
    """
    Instantiate the dataset specified by its
    key in the registry and its keyword arguments
    """
    if key not in k_dataset_registry:
        log.ERROR(f"The specified dataset <{key}> is not registered. Available datasets: {list(k_dataset_registry.keys())}")
        return None
    return k_dataset_registry[key](**kwargs)