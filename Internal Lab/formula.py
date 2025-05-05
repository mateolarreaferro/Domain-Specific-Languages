# formula.py
from typing import Self, Callable, List, TypeVar, Optional, Dict, Tuple
import functools
import pycosat
import inspect



# Core AST node
class Term:
    children: List[Self]

    def __init__(self, *children: List[Self]):
        self.children = list(children)

    # textual representation for debugging / doctest
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(repr(c) for c in self.children)})"

    # 1. Re‑express the term using only NANDs
    def to_nands(self) -> Self:
        """Rewrite this term using only NANDs (recursively)."""
        cache: Dict["Term", "Term"] = {}

        def convert(t: "Term") -> "Term":
            if t in cache:
                return cache[t]

            # Variables stay as‑is
            if isinstance(t, Var):
                cache[t] = t
                return t

            # Already a (possibly n‑ary) NAND
            if isinstance(t, Nand):
                new_node = Nand(*(convert(c) for c in t.children))
                cache[t] = new_node
                return new_node

            # Map each logical operator to its NAND‑only expression
            if isinstance(t, Not):
                a = convert(t.children[0])
                # ¬a  ≡  NAND(a, a)
                new_node = Nand(a, a)

            elif isinstance(t, And):
                a, b = (convert(c) for c in t.children)
                # (a ∧ b) ≡ NAND(NAND(a, b), NAND(a, b))
                nab = Nand(a, b)
                new_node = Nand(nab, nab)

            elif isinstance(t, Or):
                a, b = (convert(c) for c in t.children)
                # (a ∨ b) ≡ NAND(NAND(a, a), NAND(b, b))
                na = Nand(a, a)
                nb = Nand(b, b)
                new_node = Nand(na, nb)

            elif isinstance(t, Xor):
                a, b = (convert(c) for c in t.children)
                # XOR via NANDs:
                # a⊕b ≡ NAND(NAND(a, NAND(a, b)), NAND(b, NAND(a, b)))
                nab = Nand(a, b)
                left = Nand(a, nab)
                right = Nand(b, nab)
                new_node = Nand(left, right)

            else:
                raise TypeError(f"Unhandled Term subtype: {type(t)}")

            cache[t] = new_node
            return new_node

        return convert(self)

    
    # 2. Convert to CNF (Tseitin on top of NAND form)
    def to_cnf(self) -> Tuple[List[List[int]], Dict[str, int]]:
        """
        Translate the formula to an *equisatisfiable* CNF suitable for pycosat.
        Uses a Tseitin–style encoding where each sub‑term (after NAND rewrite)
        is given its own fresh CNF variable.
        """
        root = self.to_nands()                      # work only with NAND / Var
        clauses: List[List[int]] = []               # final CNF
        var_map: Dict[str, int] = {}                # Var‑name → CNF var id
        term_var_cache: Dict["Term", int] = {}      # any Term → CNF var id
        next_cnf_var = 1                            # counter for fresh vars

        def fresh_var() -> int:
            nonlocal next_cnf_var
            v = next_cnf_var
            next_cnf_var += 1
            return v


        # Recursively encode each term, return its CNF variable id
        def encode(t: "Term") -> int:
            # Cached?
            if t in term_var_cache:
                return term_var_cache[t]

            # Atomic formula variable
            if isinstance(t, Var):
                if t.name not in var_map:
                    var_map[t.name] = fresh_var()
                term_var_cache[t] = var_map[t.name]
                return var_map[t.name]

            # NAND node
            if isinstance(t, Nand):
                # Ensure all children are encoded first
                child_vars = [encode(c) for c in t.children]
                y = fresh_var()                     # CNF var for this NAND
                term_var_cache[t] = y

                # (¬x1 ∨ … ∨ ¬xn ∨ ¬y)
                clauses.append([-cv for cv in child_vars] + [-y])

                # (y ∨ xi)  for each xi
                for cv in child_vars:
                    clauses.append([y, cv])

                return y

            raise TypeError(f"Unexpected Term type: {type(t)}")

        root_cnf_var = encode(root)

        # Assert root is true
        clauses.append([root_cnf_var])

        return clauses, var_map


    # 3. Solve via pycosat
    def solve(self) -> Optional[Dict[str, bool]]:
        """Solve this formula using pycosat."""
        clauses, var_map = self.to_cnf()
        solution = pycosat.solve(clauses)

        if solution == "UNSAT":
            return None   # no satisfying assignment
        assert solution != "UNKNOWN", "pycosat returned UNKNOWN"

        # pycosat returns a list like [1, -2, 3, …]
        return {name: (solution[idx - 1] > 0) for name, idx in var_map.items()}


    # 4. Operator overloading so users can write x & ~y | z ^ w
    # '&'  → And
    def __and__(self, other: "Term") -> "Term":
        return And(self, other)

    # '|'  → Or
    def __or__(self, other: "Term") -> "Term":
        return Or(self, other)

    # '^'  → Xor
    def __xor__(self, other: "Term") -> "Term":
        return Xor(self, other)

    # '~'  → Not
    def __invert__(self) -> "Term":
        return Not(self)

    # Right‑hand variants (&, |, ^) so literals on right side also work
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__



# Concrete AST subclasses (empty shells; behaviour lives in Term)
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
    """Leaf node representing a formula variable."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self) -> str:  # keep nice printing :)
        return f"Var({self.name!r})"



# DSL entry point 
def sat(f: Callable) -> Optional[Dict[str, bool]]:
    """
    Evaluate a Python lambda / function as a propositional formula
    and return a satisfying assignment (or None if UNSAT).
    """
    # 1. Create Var objects for each formal parameter
    param_names = list(inspect.signature(f).parameters)
    vars_ = [Var(name) for name in param_names]

    # 2. Build the term by calling the user function with those Vars
    term = f(*vars_)

    # 3. Solve and return
    return term.solve()
