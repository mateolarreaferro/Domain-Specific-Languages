from timeme import timeme
from memoize import memoize


def sad_fib(n: int) -> int:
    if n < 2:
        return n
    else:
        return sad_fib(n - 1) + sad_fib(n - 2)


@memoize
def fib(n: int) -> int:
    if n < 2:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


n = 36
with timeme(f"sad_fib({n})"):
    sad_fib(n)  # seconds

with timeme(f"fib({n})"):
    fib(n)  # 10s of microseconds

assert len(fib.cache) == n + 1


@memoize
def binary_tree_count(n_leaves: int) -> int:
    if n <= 2:
        return 1
    else:
        return sum(
            binary_tree_count(n_leaves - 1 - i) * binary_tree_count(i)
            for i in range(1, n_leaves - 1)
        )


with timeme(f"binary_tree_count({n})"):
    binary_tree_count(n)  # 100s of microseconds
