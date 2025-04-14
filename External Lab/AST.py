from enum import Enum
from utils import ScopedDict
import numpy as np
import pickle
from dataclasses import dataclass


class BinOp(Enum):
    """Enum representing each arithmetic operator"""

    PLUS = 1
    MINUS = 2
    TIMES = 3


class Statement:
    """Statement base class"""

    pass


class Expr(Statement):
    """Expression base class"""

    pass


@dataclass
class Block:
    """A "block" of code, which consists of a list of statements.

    This is always the root node of the AST.
    """

    stmts: list[Statement]

    def __eq__(self, other):
        return type(self) is type(other) and self.stmts == other.stmts


@dataclass
class BinaryExpr(Expr):
    """A binary expression, consisting of a left operand,
    right operand, and operator
    """

    left: Expr
    right: Expr
    op: BinOp

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.left == other.left
            and self.right == other.right
            and self.op == other.op
        )


@dataclass
class Let(Statement):
    """A let statement"""

    name: str
    value: Expr

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.value == other.value
        )


@dataclass
class Literal(Expr):
    """An integer literal, e.g. 5"""

    value: int

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value


@dataclass
class Variable(Expr):
    """A variable"""

    name: str

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name


@dataclass
class MatrixLiteral(Expr):
    """A matrix literal, e.g. [[1, 2], [3, 4]]"""

    values: list[list[Expr]]

    def __eq__(self, other):
        return type(self) is type(other) and self.values == other.values


@dataclass
class FunctionCall(Expr):
    """A function call"""

    name: str
    args: list[Expr]

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.args == other.args
        )


@dataclass
class FunctionDec(Statement):
    """A function declaration"""

    name: str
    params: list[str]
    body: Block
    ty: "FunctionType"

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.params == other.params
            and self.body == other.body
            and self.ty == other.ty
        )


@dataclass
class Return(Statement):
    """A return statement"""

    expr: Expr

    def __eq__(self, other):
        return type(self) is type(other) and self.expr == other.expr


def interpret_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict):
    """Interpret an expression. Bindings is a dicationary that maps
    variable names to values, and declarations is a dictionary that
    maps function names to their AST nodes.

    Returns the value the expression evaluates to.
    """
    pass  # TODO (we have ~54 lines)


def interpret_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """Interpret a statment.

    Returns (value, is_ret), where value is the return value of the statement
        (or [] if there is none), and is_ret is true if the statement is a
        Return statement
    """
    pass  # TODO (we have ~15 lines)


def interpret_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    """Interpret each statement in the block.

    Returns the return value of the block, or [] if there is None
    """
    pass  # TODO (we have ~5 lines)
