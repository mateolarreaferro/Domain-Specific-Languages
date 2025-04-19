from enum import Enum
from utils import ScopedDict     # Utility for managing variable/function scopes
import numpy as np             # Used for matrix arithmetic
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
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.elements == other.elements)

@dataclass
class MatrixLiteral(Expr):
    values: list[list[Expr]]
    node_type: str = "MatrixLiteral"
    
    def __eq__(self, other):
        return (type(self) is type(other) and 
                self.values == other.values)

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
    args: list[Expr]
    def __eq__(self, other):
        return type(self) is type(other) and self.args == other.args


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
# Interpreter Helper Functions
# ----------------------------------------------------------------

def _to_matrix(val):
    """Accept int | np.ndarray and return a 2‑D numpy array."""
    if isinstance(val, np.ndarray):
        return val
    return np.array([[val]])

# ----------------------------------------------------------------
# Interpreter Functions
# ----------------------------------------------------------------

def interpret_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict):
    """
    Interpret an expression, returning its computed value.
    
    Parameters:
        expr (Expr): Expression to interpret
        bindings (ScopedDict): Variable bindings
        declarations (ScopedDict): Function declarations
        
    Returns:
        numpy.ndarray: Value as a matrix
    """
    if isinstance(expr, Literal):
        return _to_matrix(expr.value)
    
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
    
    elif isinstance(expr, VectorLiteral):
        elems = [interpret_expr(e, bindings, declarations) for e in expr.elements]
        flat = [int(v) if v.size == 1 else v.item() for v in elems]
        return np.array([flat])
    
    # Update the function call evaluation in interpret_expr
    elif isinstance(expr, FunctionCall):
        # 1) Handle built-in print function
        if expr.name == "print":
            arg_vals = [interpret_expr(a, bindings, declarations) for a in expr.args]
            for v in arg_vals:
                print(v, end=" ")
            print()
            return np.empty((0, 0))  # Mat(0,0) - spec says print returns empty matrix
        
        # 2) Handle user-defined functions
        if expr.name not in declarations:
            raise Exception(f"Undefined function: {expr.name}")
        
        func_dec = declarations[expr.name]
        arg_values = [interpret_expr(arg, bindings, declarations) for arg in expr.args]
        
        # Check if arguments match parameters count
        if len(arg_values) != len(func_dec.params):
            raise Exception(f"Function '{func_dec.name}' expected {len(func_dec.params)} arguments, got {len(arg_values)}")
        
        # Create a new scope for function execution
        func_bindings = ScopedDict()
        func_bindings.push_scope()
        
        # Bind parameters to argument values
        for param_name, arg_val in zip(func_dec.params, arg_values):
            func_bindings[param_name] = arg_val
        
        # Interpret the function body with the new bindings
        return interpret_block(func_dec.body, func_bindings, declarations)
    else:
        raise Exception(f"Unrecognized expression type: {type(expr)}")

def interpret_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """
    Interpret a statement, updating bindings and returning any value produced.
    
    Parameters:
        stmt (Statement): Statement to interpret
        bindings (ScopedDict): Variable bindings
        declarations (ScopedDict): Function declarations
        
    Returns:
        tuple: (value, is_return_value)
            value: The value produced by the statement, or None
            is_return_value: True if this was a return statement, otherwise False
    """
    # 1. let statement
    if isinstance(stmt, Let):
        val = interpret_expr(stmt.value, bindings, declarations)
        bindings[stmt.name] = val
        return val, False
    
    # 2. return statement
    if isinstance(stmt, Return):
        val = interpret_expr(stmt.expr, bindings, declarations)
        return val, True
    
    # 3. function declaration
    if isinstance(stmt, FunctionDec):
        declarations[stmt.name] = stmt
        return None, False
    
    # 4. print statement (built-in)
    if isinstance(stmt, Print):
        for arg in stmt.args:
            val = interpret_expr(arg, bindings, declarations)
            print(val, end=" ")
        print()
        return np.empty((0, 0)), False  # Mat(0,0)
    
    # 5. expression statement
    # In interpret_stmt, update the ExpressionStatement case
    if isinstance(stmt, ExpressionStatement):
        val = interpret_expr(stmt.expr, bindings, declarations)
        return val, False
    # 6. any expression used directly as a statement (rare)
    if isinstance(stmt, Expr):
        interpret_expr(stmt, bindings, declarations)
        return np.empty((0, 0)), False
    
    raise Exception(f"Unknown statement type: {type(stmt)}")

def interpret_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    """
    Interpret a block of statements, returning the final result.
    
    Parameters:
        block (Block): Block to interpret
        bindings (ScopedDict): Variable bindings
        declarations (ScopedDict): Function declarations
        
    Returns:
        numpy.ndarray: Final value produced by the block
    """
    result = np.empty((0, 0))  # Mat(0,0) default
    
    for stmt in block.stmts:
        val, is_ret = interpret_stmt(stmt, bindings, declarations)
        
        # If return statement, immediately return its value
        if is_ret:
            return val
        
        # Otherwise, track the most recent non-None value
        if val is not None:
            result = val
    
    return result