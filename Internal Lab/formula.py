from typing import Self, Callable, List, TypeVar, Optional, Dict, Tuple
import functools
import pycosat
import inspect


class Term:
    children: List[Self]

    def __init__(self, *children: List[Self]):
        self.children = children

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(repr(c) for c in self.children)})"

     # (we have ~7 lines of helper code here)

    def to_nands(self) -> Self:
        """Rewrite this term using only NANDs."""

        # hint: keep a cache of rewritten terms: (original Term) -> (only-NANDs Term)
        # (this avoids rewriting the same sub-term Term twice)

        pass  # TODO (we have ~25 lines)

    def to_cnf(self) -> Tuple[List[List[int]], Dict[str, int]]:
        """Convert a formula (self) to an equisatisfiable CNF.

        Output encoding for a CNF over N variables:
          * a variable is an integer in [1, N]
          * a literal is a positive or negative variable, representing the variable or its negation.
          * a clause is a list of literals, representing their disjunction
          * a CNF if a list of clauses, representing their conjunction

        Returns:
          * A CNF
          * a map from formula variable names to the CNF variable that encodes them
            * does not contain all CNF vars: some correspond to intermediate terms---not formula variables
        """
        self_as_nands = self.to_nands()
        clauses = []  # (add new clauses to this)
        var_map = {}  # map: var name -> cnf var (add one entry per formula var)
        next_cnf_var = 1  # (the next unused CNF var; increment as you go)

        # hint: again, keep a cache: Term -> CNF var that represents it
        # (this avoids translating the same sub-term Term twice)

        pass  # TODO (we have ~30 lines)

        return (clauses, var_map)

    def solve(self) -> Optional[Dict[str, bool]]:
        """Solve this formula using pycosat."""
        clauses, var_map = self.to_cnf()
        assert isinstance(clauses, list)
        assert len(clauses) == 0 or isinstance(clauses[0], list)
        assert len(clauses[0]) == 0 or isinstance(clauses[0][0], int)
        assert isinstance(var_map, dict)
        solution = pycosat.solve(clauses)
        assert solution != "UNKNOWN"
        if solution == "UNSAT":
            return None
        else:
            # solution is list [(+/-)1, (+/-)2, ...]
            return {
                var_name: solution[cnf_var - 1] > 0
                for var_name, cnf_var in var_map.items()
            }

    # operator overloading: (every operator but NAND)
    pass  # TODO (we have ~12 lines)


# subclasses for Term types
class Nand(Term):
    pass


class And(Term):
    pass


class Or(Term):
    pass


class Not(Term):
    pass


class Xor(Term):
    pass


class Var(Term):
    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        return f"Var({repr(self.name)})"


def sat(f) -> Optional[Dict[str, bool]]:
    """Find a satisfying assignment to this (boolean) function; if possible"""
    # hint: use `inspect.signature(f).parameters`
    pass  # TODO (we have ~3 lines)
