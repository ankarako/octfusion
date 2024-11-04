import utils.log as log

__all__ = ['register_trainer', 'get_trainer']

k_trainer_registry = { }

def register_trainer(cls):
    """
    Decorator for registering trainers in
    the trainer registry.
    """
    k_trainer_registry[cls.__name__] = cls
    return cls

def get_trainer(key: str, **kwargs):
    """
    Instantiate the trainer specified by its key
    in the registry and the specified keyword arguments
    """
    if key not in k_trainer_registry:
        log.ERROR(f"Invalid trainer. Expected one of {list(k_trainer_registry.keys())}, got {key}.")
        return None
    return k_trainer_registry[key](**kwargs)