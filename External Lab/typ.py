from AST import *            # Import AST classes (e.g., Literal, Variable, BinaryExpr, Let, etc.)
from dataclasses import dataclass
from utils import ScopedDict

# --- Function to clone a scoped dictionary for type checking -----------------
def _clone_scoped_dict(src: ScopedDict) -> ScopedDict:
    new = ScopedDict()                 # start with one empty scope
    new.dicts = [d.copy() for d in src.dicts]  # deep‑copy each scope
    return new

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
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"ConcreteDim({self.value})"

# ----------------------------------------------------------------
# Type variable dimension: represents a polymorphic dimension (e.g., 'a', 'b', 'c')
# ----------------------------------------------------------------
@dataclass
class TypeVarDim(Dim):
    """Type variable dimension for polymorphic functions."""
    name: str

    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"TypeVarDim({self.name})"

# ----------------------------------------------------------------
# MatrixType: represents a matrix type as Mat(r, c)
# ----------------------------------------------------------------
@dataclass
class MatrixType(Type):
    """Type for a matrix with a shape given by (number of rows, number of columns)."""
    # shape is a tuple (Dim, Dim)
    shape: tuple

    def __eq__(self, other):
        # Two matrix types are equal if their shapes (dimensions) are equal.
        return type(self) is type(other) and self.shape == other.shape
    
    def __str__(self):
        rows, cols = self.shape
        return f"Mat({rows}, {cols})"
    
    def __repr__(self):
        return f"MatrixType(shape=({self.shape[0]}, {self.shape[1]}))"

# ----------------------------------------------------------------
# FunctionType: represents a function's type signature
# ----------------------------------------------------------------
@dataclass
class FunctionType(Type):
    """Type of a function, including parameter types and a return type."""
    params: list[Type]  # List of parameter types.
    ret: Type           # Return type of the function.

    def __eq__(self, other):
        # Function types are equal if their parameter lists and return types are equal.
        return (type(self) is type(other)
                and self.params == other.params
                and self.ret == other.ret)
    
    def __str__(self):
        param_strs = [str(p) for p in self.params]
        return f"({', '.join(param_strs)}) -> {self.ret}"
    
    def __repr__(self):
        return f"FunctionType(params={self.params}, ret={self.ret})"

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
    # Integer literal: Mat(1,1)
    if isinstance(expr, Literal):
        return MatrixType((ConcreteDim(1), ConcreteDim(1)))
    
    # Variable reference: look up in bindings
    elif isinstance(expr, Variable):
        if bindings.get(expr.name) is None:
            raise TypeError(f"Undefined variable: {expr.name}")
        return bindings[expr.name]
    
    # Vector literal: check all elements are scalars, return Mat(1,n)
    elif isinstance(expr, VectorLiteral):
        for e in expr.elements:
            etype = type_expr(e, bindings, declarations)
            if not isinstance(etype, MatrixType) or etype.shape != (ConcreteDim(1), ConcreteDim(1)):
                raise TypeError("Vector elements must be Mat(1,1)")
        return MatrixType((ConcreteDim(1), ConcreteDim(len(expr.elements))))
    
    # Binary expression: +, -, *
    elif isinstance(expr, BinaryExpr):
        left_type = type_expr(expr.left, bindings, declarations)
        right_type = type_expr(expr.right, bindings, declarations)
        
        # Ensure both operands are matrix types
        if not isinstance(left_type, MatrixType) or not isinstance(right_type, MatrixType):
            raise TypeError("Binary operators require both operands to be matrices.")
        
        # Addition/subtraction: dimensions must match exactly
        if expr.op in (BinOp.PLUS, BinOp.MINUS):
            if left_type.shape != right_type.shape:
                raise TypeError(f"Operands of '+' or '-' must have the same dimensions: {left_type} vs {right_type}")
            return left_type
        
        # Matrix multiplication: left cols must equal right rows
        elif expr.op == BinOp.TIMES:
            left_rows, left_cols = left_type.shape
            right_rows, right_cols = right_type.shape
            
            if not(left_cols == right_rows):
                raise TypeError(f"For matrix multiplication, left matrix's columns must equal right matrix's rows: {left_type} * {right_type}")
            
            # Result type: (left_rows, right_cols)
            return MatrixType((left_rows, right_cols))
        
        else:
            raise TypeError(f"Unknown binary operator: {expr.op}")
    
    # Matrix literal
    elif isinstance(expr, MatrixLiteral):
        # Empty matrix: Mat(0,0)
        if len(expr.values) == 0:
            return MatrixType((ConcreteDim(0), ConcreteDim(0)))
        
        # All rows must have same length
        row_length = len(expr.values[0])
        for row in expr.values:
            if len(row) != row_length:
                raise TypeError("All rows in a matrix literal must have the same number of elements.")
            
            # All elements must be scalars (Mat(1,1))
            for elem in row:
                elem_type = type_expr(elem, bindings, declarations)
                if not (isinstance(elem_type, MatrixType) and elem_type.shape == (ConcreteDim(1), ConcreteDim(1))):
                    raise TypeError("Elements of a matrix literal must be integer literals.")
        
        # Return Mat(rows, cols)
        return MatrixType((ConcreteDim(len(expr.values)), ConcreteDim(row_length)))
    
    # Function call
    elif isinstance(expr, FunctionCall):
        # Special case for built-in print function
        if expr.name == "print":
            # Check that all arguments are well-typed
            for arg in expr.args:
                type_expr(arg, bindings, declarations)
            # print returns Mat(0,0)
            return MatrixType((ConcreteDim(0), ConcreteDim(0)))
        
        # User-defined function
        if expr.name not in declarations:
            raise TypeError(f"Undefined function: {expr.name}")
        
        func_decl = declarations[expr.name]
        func_type = func_decl.ty
        
        if not isinstance(func_type, FunctionType):
            raise TypeError("Invalid function type encountered.")
        
        # The number of arguments must match the number of parameters
        if len(expr.args) != len(func_type.params):
            raise TypeError(f"Function call argument count mismatch: expected {len(func_type.params)}, got {len(expr.args)}.")
        
        # Check for polymorphic function handling
        type_vars = {}  # Maps type variable names to concrete dimensions
        
        # For each argument, check against parameter type
        for arg_idx, (arg_expr, param_type) in enumerate(zip(expr.args, func_type.params)):
            arg_type = type_expr(arg_expr, bindings, declarations)
            
            # Both must be matrix types
            if not isinstance(arg_type, MatrixType) or not isinstance(param_type, MatrixType):
                raise TypeError(f"Function argument and parameter types must be matrix types")
            
            # Check row dimensions
            arg_rows, arg_cols = arg_type.shape
            param_rows, param_cols = param_type.shape
            
            # Handle polymorphic row dimension
            if isinstance(param_rows, TypeVarDim):
                var_name = param_rows.name
                if var_name in type_vars:
                    # Type variable already bound, must match
                    if type_vars[var_name] != arg_rows:
                        raise TypeError(f"Type variable '{var_name}' was bound to {type_vars[var_name]} but got {arg_rows} in argument {arg_idx+1}")
                else:
                    # Bind type variable to concrete dimension
                    type_vars[var_name] = arg_rows
            elif arg_rows != param_rows:
                raise TypeError(f"Argument {arg_idx+1} has incompatible row dimension: expected {param_rows}, got {arg_rows}")
            
            # Handle polymorphic column dimension
            if isinstance(param_cols, TypeVarDim):
                var_name = param_cols.name
                if var_name in type_vars:
                    # Type variable already bound, must match
                    if type_vars[var_name] != arg_cols:
                        raise TypeError(f"Type variable '{var_name}' was bound to {type_vars[var_name]} but got {arg_cols} in argument {arg_idx+1}")
                else:
                    # Bind type variable to concrete dimension
                    type_vars[var_name] = arg_cols
            elif arg_cols != param_cols:
                raise TypeError(f"Argument {arg_idx+1} has incompatible column dimension: expected {param_cols}, got {arg_cols}")
        
        # Substitute type variables in return type
        ret_type = func_type.ret
        if isinstance(ret_type, MatrixType):
            ret_rows, ret_cols = ret_type.shape
            
            if isinstance(ret_rows, TypeVarDim) and ret_rows.name in type_vars:
                ret_rows = type_vars[ret_rows.name]
            
            if isinstance(ret_cols, TypeVarDim) and ret_cols.name in type_vars:
                ret_cols = type_vars[ret_cols.name]
            
            return MatrixType((ret_rows, ret_cols))
        
        return ret_type
    
    # Unrecognized expression type
    else:
        raise TypeError(f"Unrecognized expression type: {type(expr)}")

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
    # Variable binding with let
    if isinstance(stmt, Let):
        t = type_expr(stmt.value, bindings, declarations)
        bindings[stmt.name] = t
    
    # Return statement
    elif isinstance(stmt, Return):
        type_expr(stmt.expr, bindings, declarations)
    
    # Function declaration
    elif isinstance(stmt, FunctionDec):
        # 1. Register the function so later calls see its type
        declarations[stmt.name] = stmt

        # 2. Prepare an inner scope holding the parameter bindings
        inner_bindings = _clone_scoped_dict(bindings)
        inner_bindings.push_scope()  
        for p_name, p_type in zip(stmt.params, stmt.ty.params):
            inner_bindings[p_name] = p_type

        # 3. Type‑check the body
        #    Track every explicit return so we can verify its type.
        observed_return_types = []

        for s in stmt.body.stmts:
            if isinstance(s, Return):
                observed_return_types.append(type_expr(s.expr, inner_bindings, declarations))
            else:
                type_stmt(s, inner_bindings, declarations)

        # 4. If the function has explicit returns, make sure each matches the declared type
        if observed_return_types:
            for rt in observed_return_types:
                if rt != stmt.ty.ret:
                    raise TypeError(
                        f"Return type mismatch in function '{stmt.name}': "
                        f"expected {stmt.ty.ret} but got {rt}"
                    )
        # 5. If there is **no** return statement, spec says it returns Mat(0,0)
        else:
            implicit = MatrixType((ConcreteDim(0), ConcreteDim(0)))
            if implicit != stmt.ty.ret:
                raise TypeError(
                    f"Function '{stmt.name}' is missing a return; "
                    f"declared return type was {stmt.ty.ret}"
                )
    
    # Print statement
    elif isinstance(stmt, Print):
        for arg in stmt.args:
            type_expr(arg, bindings, declarations)
    
    # Expression statement
    elif isinstance(stmt, ExpressionStatement): 
        type_expr(stmt.expr, bindings, declarations)
    
    # Expression used as statement
    elif isinstance(stmt, Expr):
        type_expr(stmt, bindings, declarations)
    
    # Unknown statement type
    else:
        raise TypeError(f"Unknown statement type: {type(stmt)}")

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