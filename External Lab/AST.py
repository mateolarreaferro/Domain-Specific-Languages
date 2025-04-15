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
# AST Class: Variable (for variable references)
# ----------------------------------------------------------------
@dataclass
class Variable(Expr):
    """A variable reference."""
    name: str

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

# ----------------------------------------------------------------
# AST Class: MatrixLiteral (for matrix literals)
# ----------------------------------------------------------------
@dataclass
class MatrixLiteral(Expr):
    """A matrix literal (e.g., [[1, 2], [3, 4]]). 
    'values' is a list of rows, where each row is a list of expressions.
    """
    values: list[list[Expr]]

    def __eq__(self, other):
        return type(self) is type(other) and self.values == other.values

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
    """
    Evaluate an expression 'expr'.
    
    - 'bindings' is a dictionary mapping variable names to their numpy array values.
    - 'declarations' maps function names to their AST nodes.
    
    Returns:
       A numpy array representing the matrix result of evaluating the expression.
    """
    # Evaluate an integer literal as a 1x1 matrix.
    if isinstance(expr, Literal):
        # Return a numpy array with one row and one column.
        return np.array([[expr.value]])
    
    # Evaluate a variable by looking it up in the bindings.
    elif isinstance(expr, Variable):
        if expr.name not in bindings:
            raise Exception(f"Undefined variable: {expr.name}")
        return bindings[expr.name]
    
    # Evaluate a binary expression by evaluating the left and right operands and applying the operator.
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
    
    # Evaluate a matrix literal by evaluating each row.
    elif isinstance(expr, MatrixLiteral):
        evaluated_rows = []
        for row in expr.values:
            # Each row is a list of expressions (expected to be literals).
            evaluated_row = [interpret_expr(e, bindings, declarations)[0][0] for e in row]
            evaluated_rows.append(evaluated_row)
        # Convert the list of rows into a numpy array.
        return np.array(evaluated_rows)
    
    # Evaluate a function call by evaluating its arguments and invoking the function.
    elif isinstance(expr, FunctionCall):
        if expr.name not in declarations:
            raise Exception(f"Undefined function: {expr.name}")
        func_dec = declarations[expr.name]
        # Evaluate each argument.
        arg_values = [interpret_expr(arg, bindings, declarations) for arg in expr.args]
        # Create a new scope that inherits from 'bindings' to bind function parameters.
        new_bindings = ScopedDict(bindings)
        # Bind each parameter to the corresponding argument value.
        for param_name, arg_val in zip(func_dec.params, arg_values):
            new_bindings[param_name] = arg_val
        # Evaluate the function body (a Block) using the new bindings.
        return interpret_block(func_dec.body, new_bindings, declarations)
    
    else:
        raise Exception("Unrecognized expression type: " + str(type(expr)))


def interpret_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """
    Evaluate a statement.
    
    Returns:
       (value, is_ret) where:
         - 'value' is the result of the statement (or an empty structure if none),
         - 'is_ret' is a boolean that is True if the statement is a Return statement.
    """
    # For a let statement, evaluate the value, bind it, and return the value.
    if isinstance(stmt, Let):
        val = interpret_expr(stmt.value, bindings, declarations)
        bindings[stmt.name] = val
        return (val, False)
    
    # For a return statement, evaluate the expression and indicate a return was hit.
    elif isinstance(stmt, Return):
        val = interpret_expr(stmt.expr, bindings, declarations)
        return (val, True)
    
    # For a function declaration, add it to the declarations environment.
    elif isinstance(stmt, FunctionDec):
        declarations[stmt.name] = stmt
        return (None, False)
    
    # For an expression statement, simply evaluate the expression.
    elif isinstance(stmt, Expr):
        val = interpret_expr(stmt, bindings, declarations)
        return (val, False)
    
    else:
        raise Exception("Unknown statement type: " + str(type(stmt)))


def interpret_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    """
    Evaluate each statement in a block sequentially.
    
    Returns:
       The value from a return statement if encountered; otherwise, returns an empty matrix.
    """
    for stmt in block.stmts:
        val, is_ret = interpret_stmt(stmt, bindings, declarations)
        if is_ret:
            return val
    # If no return is encountered in the block, return an empty matrix.
    return np.array([[]])
