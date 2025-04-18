from enum import Enum
from utils import ScopedDict     # Utility for managing variable/function scopes
import numpy as np             # Used for matrix arithmetic
import pickle
from dataclasses import dataclass

# ----------------------------------------------------------------
# Binary Operator Enum
# ----------------------------------------------------------------
class BinOp(Enum):
    """Enum representing each arithmetic operator."""
    PLUS = 1
    MINUS = 2
    TIMES = 3

# ----------------------------------------------------------------
# Base classes for Statements and Expressions
# ----------------------------------------------------------------
class Statement:
    """Base class for all statements in the AST."""
    pass

class Expr(Statement):
    """Base class for all expressions in the AST."""
    pass

@dataclass
class ExpressionStatement(Statement):
    expr: Expr
    def __eq__(self, other):
        return type(self) is type(other) and self.expr == other.expr


# ----------------------------------------------------------------
# AST Class: Block
# ----------------------------------------------------------------
@dataclass
class Block:
    """A block of code consisting of a list of statements.

    This is always the root node of the AST.
    """
    stmts: list[Statement]

    def __eq__(self, other):
        return type(self) is type(other) and self.stmts == other.stmts

# ----------------------------------------------------------------
# AST Class: BinaryExpr (for binary arithmetic expressions)
# ----------------------------------------------------------------
@dataclass
class BinaryExpr(Expr):
    """A binary expression with a left operand, a right operand, and an operator."""
    left: Expr
    right: Expr
    op: BinOp

    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.left == other.left and 
                self.right == other.right and 
                self.op == other.op)

@dataclass
class VectorLiteral(Expr):
    elements: list[Expr]
    node_type: str = "VectorLiteral"

@dataclass
class MatrixLiteral(Expr):
    values: list[list[Expr]]
    node_type: str = "MatrixLiteral"

# ----------------------------------------------------------------
# AST Class: Let (for variable bindings)
# ----------------------------------------------------------------
@dataclass
class Let(Statement):
    """A let statement that binds an expression to a variable."""
    name: str
    value: Expr

    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.name == other.name and 
                self.value == other.value)

# ----------------------------------------------------------------
# AST Class: Literal (for integer literals)
# ----------------------------------------------------------------
@dataclass
class Literal(Expr):
    """An integer literal (e.g., 5)."""
    value: int

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value
    
# ----------------------------------------------------------------
# AST Class: Print (for print statements)
# ----------------------------------------------------------------
@dataclass
class Print(Statement):
    expr: Expr

    def __eq__(self, other):
        return type(self) is type(other) and self.expr == other.expr


# ----------------------------------------------------------------
# AST Class: Variable (for variable references)
# ----------------------------------------------------------------
@dataclass
class Variable(Expr):
    """A variable reference."""
    name: str

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

# ----------------------------------------------------------------
# AST Class: FunctionCall (for function calls)
# ----------------------------------------------------------------
@dataclass
class FunctionCall(Expr):
    """A function call with a name and a list of argument expressions."""
    name: str
    args: list[Expr]

    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.name == other.name and 
                self.args == other.args)

# ----------------------------------------------------------------
# AST Class: FunctionDec (for function declarations/definitions)
# ----------------------------------------------------------------
@dataclass
class FunctionDec(Statement):
    """A function declaration with a name, parameter names, a body (Block), 
    and a type annotation for its signature.
    """
    name: str
    params: list[str]
    body: Block
    ty: "FunctionType"    # A FunctionType (defined elsewhere or as needed)

    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.name == other.name and 
                self.params == other.params and 
                self.body == other.body and 
                self.ty == other.ty)

# ----------------------------------------------------------------
# AST Class: Return (for return statements)
# ----------------------------------------------------------------
@dataclass
class Return(Statement):
    """A return statement that evaluates and returns an expression."""
    expr: Expr

    def __eq__(self, other):
        return type(self) is type(other) and self.expr == other.expr

# ----------------------------------------------------------------
# Interpreter Functions
# ----------------------------------------------------------------

def interpret_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict):
    if isinstance(expr, Literal):
        return np.array([[expr.value]])
    elif isinstance(expr, Variable):
        if expr.name not in bindings:
            raise Exception(f"Undefined variable: {expr.name}")
        return bindings[expr.name]
    elif isinstance(expr, BinaryExpr):
        left_val = interpret_expr(expr.left, bindings, declarations)
        right_val = interpret_expr(expr.right, bindings, declarations)
        if expr.op == BinOp.PLUS:
            return left_val + right_val
        elif expr.op == BinOp.MINUS:
            return left_val - right_val
        elif expr.op == BinOp.TIMES:
            return np.matmul(left_val, right_val)
        else:
            raise Exception(f"Unknown binary operator: {expr.op}")
    elif isinstance(expr, MatrixLiteral):
        evaluated_rows = []
        for row in expr.values:
            evaluated_row = [interpret_expr(e, bindings, declarations)[0][0] for e in row]
            evaluated_rows.append(evaluated_row)
        return np.array(evaluated_rows)
    elif isinstance(expr, FunctionCall):
        if expr.name not in declarations:
            raise Exception(f"Undefined function: {expr.name}")
        func_dec = declarations[expr.name]
        arg_values = [interpret_expr(arg, bindings, declarations) for arg in expr.args]
        new_bindings = ScopedDict(bindings)
        for param_name, arg_val in zip(func_dec.params, arg_values):
            new_bindings[param_name] = arg_val
        return interpret_block(func_dec.body, new_bindings, declarations)
    elif isinstance(expr, VectorLiteral):
        row = [interpret_expr(e, bindings, declarations)[0][0] for e in expr.elements]
        return np.array([row])
    else:
        raise Exception("Unrecognized expression type: " + str(type(expr)))

def interpret_stmt(stmt, bindings, declarations):
    if isinstance(stmt, Let):
        val = interpret_expr(stmt.expr, bindings, declarations)
        return (val, False)
    elif isinstance(stmt, Return):
        val = interpret_expr(stmt.expr, bindings, declarations)
        return (val, True)
    elif isinstance(stmt, FunctionDec):
        declarations[stmt.name] = stmt
        return (None, False)
    elif isinstance(stmt, Expr):
        val = interpret_expr(stmt, bindings, declarations)
        return (val, False)
    elif isinstance(stmt, Print):
        val = interpret_expr(stmt.expr, bindings, declarations)
        print(val)
        return (np.empty((0, 0)), False)  # Mat(0, 0)

    else:
        raise Exception("Unknown statement type: " + str(type(stmt)))

def interpret_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    for stmt in block.stmts:
        val, is_ret = interpret_stmt(stmt, bindings, declarations)
        if is_ret:
            return val
    return np.array([[]])
