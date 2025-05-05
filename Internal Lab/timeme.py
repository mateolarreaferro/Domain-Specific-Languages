from contextlib import contextmanager
from time import time


@contextmanager
def timeme(label: str):
    """Context manager that prints how long the wrapped block takes."""
    start = time()
    try:
        yield
    finally:
        elapsed = time() - start
        print(f"{label} took {elapsed:.3f} s")
