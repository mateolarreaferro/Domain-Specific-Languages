from AST import *
from dataclasses import dataclass


@dataclass
class TypeError(Exception):
    msg: str

    def __init__(self, _msg: str):
        self.msg = _msg


class Type:
    pass


class Dim:
    pass


@dataclass
class ConcreteDim(Dim):
    """Concrete dimension (e.g. a number) for a matrix"""

    value: int

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value


@dataclass
class MatrixType(Type):
    """Type for a matrix"""

    shape: (Dim, Dim)

    def __eq__(self, other):
        return type(self) is type(other) and self.shape == other.shape


@dataclass
class FunctionType(Type):
    """Type of a function"""

    params: [Type]
    ret: Type

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.params == other.params
            and self.ret == other.ret
        )


def type_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict):
    """Types an expression. bindings is a dictionary mapping variable names
    to their type. declarations is a dictionary mapping function names
    to their types.

    Returns the type of the expression.
    Raises a TypeError if the types are incorrect.
    """

    pass  # TODO (we have ~89 lines)


def type_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """Type checks a statement. bindings is a dictionary mapping variable names
    to their type. declarations is a dictionary mapping function names
    to their types.

    Raises TypeError if the types are incorrect.
    """
    pass  # TODO (we have ~32 lines)


def type_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    """Type checks a block. bindings is a dictionary mapping variable names
    to their type. declarations is a dictionary mapping function names
    to their types.

    Raises TypeError if the types are incorrect.
    """

    for statement in block.stmts:
        type_stmt(statement, bindings, declarations)
