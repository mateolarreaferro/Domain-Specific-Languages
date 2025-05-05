from functools import wraps


def memoize(func):
    """Simple memoization decorator.

    It stores results keyed by the full argument list (positional + keyword),
    attaches the cache as `func.cache`, and returns cached values on repeats.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Build a hashable key from args and kwargs
        key = (args, tuple(sorted(kwargs.items()))) if kwargs else args
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    return wrapper
