from AST import *            # Import AST classes (e.g., Literal, Variable, BinaryExpr, Let, etc.)
from dataclasses import dataclass

# ----------------------------------------------------------------
# Custom TypeError for our type checker
# ----------------------------------------------------------------
@dataclass
class TypeError(Exception):
    msg: str
    # The constructor sets the error message.
    def __init__(self, _msg: str):
        self.msg = _msg

# ----------------------------------------------------------------
# Base class for types
# ----------------------------------------------------------------
class Type:
    pass

# ----------------------------------------------------------------
# Base class for dimensions in types (e.g., the number of rows or columns)
# ----------------------------------------------------------------
class Dim:
    pass

# ----------------------------------------------------------------
# Concrete dimension: represents an actual numerical dimension (e.g., 1, 2, 3, …)
# ----------------------------------------------------------------
@dataclass
class ConcreteDim(Dim):
    """Concrete dimension (e.g. a specific number) for a matrix."""
    value: int

    def __eq__(self, other):
        # Equal if the type is the same and the contained value is equal.
        return type(self) is type(other) and self.value == other.value

# ----------------------------------------------------------------
# MatrixType: represents a matrix type as Mat(r, c)
# ----------------------------------------------------------------
@dataclass
class MatrixType(Type):
    """Type for a matrix with a shape given by (number of rows, number of columns)."""
    # shape is a tuple (Dim, Dim)
    shape: (Dim, Dim)

    def __eq__(self, other):
        # Two matrix types are equal if their shapes (dimensions) are equal.
        return type(self) is type(other) and self.shape == other.shape

# ----------------------------------------------------------------
# FunctionType: represents a function's type signature
# ----------------------------------------------------------------
@dataclass
class FunctionType(Type):
    """Type of a function, including parameter types and a return type."""
    params: [Type]  # List of parameter types.
    ret: Type       # Return type of the function.

    def __eq__(self, other):
        # Function types are equal if their parameter lists and return types are equal.
        return (type(self) is type(other)
                and self.params == other.params
                and self.ret == other.ret)

# ----------------------------------------------------------------
# Type-checking for expressions
# ----------------------------------------------------------------
def type_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict):
    """
    Type-check an expression 'expr'.
    
    Parameters:
      expr          - the AST node representing the expression.
      bindings      - a dictionary mapping variable names to their types.
      declarations  - a dictionary mapping function names to their declarations (which include their types).
    
    Returns:
      The type (an instance of MatrixType or FunctionType) of the expression.
    
    Raises:
      TypeError if the expression is ill-typed.
    """
    # If the expression is an integer literal, its type is Mat(1,1)
    if isinstance(expr, Literal):
        return MatrixType((ConcreteDim(1), ConcreteDim(1)))
    
    # If the expression is a variable, retrieve its type from the bindings.
    elif isinstance(expr, Variable):
        if expr.name not in bindings:
            raise TypeError(f"Undefined variable: {expr.name}")
        return bindings[expr.name]
    
    # If the expression is a binary expression, type-check both operands.
    elif isinstance(expr, BinaryExpr):
        left_type = type_expr(expr.left, bindings, declarations)
        right_type = type_expr(expr.right, bindings, declarations)
        # Ensure both operands are matrix types.
        if not isinstance(left_type, MatrixType) or not isinstance(right_type, MatrixType):
            raise TypeError("Binary operators require both operands to be matrices.")
        # For addition or subtraction, dimensions must be exactly equal.
        if expr.op in (BinOp.PLUS, BinOp.MINUS):
            if left_type != right_type:
                raise TypeError("Operands of '+' or '-' must have the same dimensions.")
            return left_type
        # For multiplication, the number of columns of the left must equal the number of rows of the right.
        elif expr.op == BinOp.TIMES:
            left_rows, left_cols = left_type.shape
            right_rows, right_cols = right_type.shape
            if left_cols != right_rows:
                raise TypeError("For matrix multiplication, left matrix's columns must equal right matrix's rows.")
            # The result type is a matrix with dimensions (left_rows, right_cols).
            return MatrixType((left_rows, right_cols))
        else:
            raise TypeError("Unknown binary operator encountered.")
    
    # If the expression is a matrix literal, determine its type based on its shape.
    elif isinstance(expr, MatrixLiteral):
        # An empty matrix literal is considered to have type Mat(0,0)
        if len(expr.values) == 0:
            return MatrixType((ConcreteDim(0), ConcreteDim(0)))
        # All rows must have the same number of elements.
        row_length = len(expr.values[0])
        for row in expr.values:
            if len(row) != row_length:
                raise TypeError("All rows in a matrix literal must have the same number of elements.")
            for elem in row:
                # Each element should type-check as an integer literal (Mat(1,1)).
                elem_type = type_expr(elem, bindings, declarations)
                if not (isinstance(elem_type, MatrixType) and elem_type.shape == (ConcreteDim(1), ConcreteDim(1))):
                    raise TypeError("Elements of a matrix literal must be integer literals.")
        # Return the matrix type: number of rows = len(expr.values), columns = row_length.
        return MatrixType((ConcreteDim(len(expr.values)), ConcreteDim(row_length)))
    
    # If the expression is a function call, check that the function is declared and the arguments match.
    elif isinstance(expr, FunctionCall):
        if expr.name not in declarations:
            raise TypeError(f"Undefined function: {expr.name}")
        func_decl = declarations[expr.name]
        # func_decl is assumed to be a FunctionDec with a 'ty' field of type FunctionType.
        func_type = func_decl.ty
        if not isinstance(func_type, FunctionType):
            raise TypeError("Invalid function type encountered.")
        # The number of arguments must match the number of parameters.
        if len(expr.args) != len(func_type.params):
            raise TypeError("Function call argument count mismatch.")
        # For each argument, check that its type equals the corresponding parameter type.
        for arg_expr, expected_type in zip(expr.args, func_type.params):
            arg_type = type_expr(arg_expr, bindings, declarations)
            if arg_type != expected_type:
                raise TypeError(f"Function call type mismatch: expected {expected_type} but got {arg_type}.")
        # Return the function's declared return type.
        return func_type.ret
    
    # If the expression type is unrecognized, raise an error.
    else:
        raise TypeError("Unrecognized expression type: " + str(type(expr)))

# ----------------------------------------------------------------
# Type-checking for statements
# ----------------------------------------------------------------
def type_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """
    Type-check a statement.
    
    Parameters:
      stmt          - the AST node representing the statement.
      bindings      - a dictionary mapping variable names to their types.
      declarations  - a dictionary mapping function names to their types/declarations.
    
    Raises:
      TypeError if the statement is ill-typed.
    """
    # A let statement: type-check the expression and add the variable's type into bindings.
    if isinstance(stmt, Let):
        expr_type = type_expr(stmt.value, bindings, declarations)
        bindings[stmt.name] = expr_type
    # A return statement: type-check the expression being returned.
    elif isinstance(stmt, Return):
        # We simply type-check the return expression;
        # the caller (e.g., type_stmt in a function body) should verify that the return type matches.
        type_expr(stmt.expr, bindings, declarations)
    # A function declaration: type-check the function body using a new scope.
    elif isinstance(stmt, FunctionDec):
        # Create a new binding environment for the function's scope.
        local_bindings = ScopedDict(bindings)
        # The declared function type (stmt.ty) contains parameter types.
        if len(stmt.params) != len(stmt.ty.params):
            raise TypeError("Parameter count mismatch in function declaration.")
        # Bind each parameter name to its declared type.
        for pname, ptype in zip(stmt.params, stmt.ty.params):
            local_bindings[pname] = ptype
        # Type-check the function body.
        # (If the function body returns a value, type checking of that return is done in type_expr for a Return node.)
        type_block(stmt.body, local_bindings, declarations)
        # Add the function declaration to the declarations dictionary.
        declarations[stmt.name] = stmt
    # An expression statement: simply type-check the expression.
    elif isinstance(stmt, Expr):
        type_expr(stmt, bindings, declarations)
    else:
        raise TypeError("Unknown statement type: " + str(type(stmt)))

# ----------------------------------------------------------------
# Type-checking for a block of code
# ----------------------------------------------------------------
def type_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    """
    Type-check every statement in a block.
    
    Parameters:
      block         - the Block AST node containing a list of statements.
      bindings      - a dictionary mapping variable names to their types.
      declarations  - a dictionary mapping function names to their types/declarations.
    
    Raises:
      TypeError if any statement in the block is ill-typed.
    """
    # Iterate over each statement in the block and type-check it.
    for statement in block.stmts:
        type_stmt(statement, bindings, declarations)
