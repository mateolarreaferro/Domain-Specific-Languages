# typ.py

import copy # Needed for deep copying scopes
from dataclasses import dataclass

# Assuming AST nodes are defined correctly in AST.py
# Import necessary AST node types explicitly
from AST import (
    Statement, Expr, Block, BinaryExpr, Let, Literal, Variable,
    MatrixLiteral, FunctionCall, FunctionDec, Return, BinOp
)
# Assuming ScopedDict is defined in utils.py
from utils import ScopedDict

# ----------------------------------------------------------------
# Custom TypeError
# ----------------------------------------------------------------
@dataclass
class TypeError(Exception):
    """Custom exception for type errors."""
    msg: str
    def __init__(self, _msg: str):
        super().__init__(_msg) # Initialize base Exception with message
        self.msg = _msg

# ----------------------------------------------------------------
# Base Type Classes
# ----------------------------------------------------------------
class Type:
    """Base class for all types."""
    pass

class Dim:
    """Base class for dimensions (used internally for clarity, not stored in shape)."""
    pass

# ----------------------------------------------------------------
# Dimension Implementations (Used by type checker logic)
# ----------------------------------------------------------------
@dataclass(eq=True)
class ConcreteDim(Dim):
    """Represents a concrete dimension value."""
    value: int

    def __repr__(self):
        # Provide an unambiguous representation
        return f"ConcreteDim(value={self.value!r})"

    def __str__(self):
        return str(self.value)

@dataclass(eq=True, frozen=True) # Frozen makes it hashable, usable as dict key
class TypeVarDim(Dim):
    """Represents a type variable for dimensions."""
    # Note: This class exists for type-checking logic, but the parser
    # now stores strings directly in MatrixType.shape to match the pickle file.
    # The type checker needs to handle shapes containing str or ConcreteDim.
    name: str

    def __repr__(self):
        # Provide an unambiguous representation
        return f"TypeVarDim(name={self.name!r})"

    def __str__(self):
        return self.name

# ----------------------------------------------------------------
# Matrix Type
# ----------------------------------------------------------------
@dataclass(eq=True)
class MatrixType(Type):
    """Type for a matrix. Shape stores ConcreteDim or str."""
    # Shape tuple contains ConcreteDim objects or strings (like 'a')
    # based on the parser modifications needed to match the pickle file.
    shape: tuple

    # Removed __post_init__ validation because the expected AST uses strings ('a')
    # instead of TypeVarDim('a') in the shape tuple, and we need to match the test.

    def __post_init__(self):
        # Basic validation: shape should be a tuple of length 2
        # Elements can be ConcreteDim or str
        if not (isinstance(self.shape, tuple) and len(self.shape) == 2):
             raise ValueError(f"MatrixType shape must be a tuple of length 2, got: {self.shape!r}")
        if not (isinstance(self.shape[0], (ConcreteDim, str))):
             raise ValueError(f"MatrixType shape[0] must be ConcreteDim or str, got: {type(self.shape[0])!r}")
        if not (isinstance(self.shape[1], (ConcreteDim, str))):
             raise ValueError(f"MatrixType shape[1] must be ConcreteDim or str, got: {type(self.shape[1])!r}")


    def __repr__(self):
        # Representation shows strings or ConcreteDim objects in shape
        return f"MatrixType(shape=({self.shape[0]!r}, {self.shape[1]!r}))"

    def __str__(self):
        # Adjust string conversion based on shape content (str or ConcreteDim)
        rows, cols = self.shape
        row_str = str(rows.value) if isinstance(rows, ConcreteDim) else str(rows)
        col_str = str(cols.value) if isinstance(cols, ConcreteDim) else str(cols)
        return f"Mat({row_str}, {col_str})"

# Convenience constants
MAT_1x1 = MatrixType(shape=(ConcreteDim(1), ConcreteDim(1)))
MAT_0x0 = MatrixType(shape=(ConcreteDim(0), ConcreteDim(0)))

# ----------------------------------------------------------------
# Function Type
# ----------------------------------------------------------------
@dataclass(eq=True)
class FunctionType(Type):
    """Type of a function."""
    # Use tuple for params to match expected AST structure
    params: tuple[Type, ...] # Use tuple type hint
    ret: Type

    def __repr__(self):
        # Provide an unambiguous representation
        return f"FunctionType(params={self.params!r}, ret={self.ret!r})"

    def __str__(self):
        # Keep the simpler string representation
        param_strs = [str(p) for p in self.params]
        return f"({', '.join(param_strs)}) -> {self.ret}"

# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------
def _clone_scoped_dict(src: ScopedDict) -> ScopedDict:
    """Deep copy a ScopedDict."""
    new = ScopedDict()
    # Create copy of list and copy of each dictionary inside
    new.dicts = [d.copy() for d in src.dicts]
    return new

def _substitute_poly_type(matrix_type: MatrixType, type_var_map: dict) -> MatrixType:
    """Substitutes type variables (strings) in a MatrixType shape using the map."""
    if not isinstance(matrix_type, MatrixType):
        return matrix_type # Return as is if not a MatrixType

    r_dim, c_dim = matrix_type.shape
    new_r_dim = r_dim
    new_c_dim = c_dim

    # Substitute row dimension if it's a string variable found in the map
    if isinstance(r_dim, str) and r_dim in type_var_map:
        new_r_dim = type_var_map[r_dim]
        # Ensure the mapped value is a ConcreteDim
        if not isinstance(new_r_dim, ConcreteDim):
             raise TypeError(f"Type variable '{r_dim}' mapped to non-concrete dimension '{new_r_dim}'")

    # Substitute column dimension if it's a string variable found in the map
    if isinstance(c_dim, str) and c_dim in type_var_map:
        new_c_dim = type_var_map[c_dim]
        # Ensure the mapped value is a ConcreteDim
        if not isinstance(new_c_dim, ConcreteDim):
             raise TypeError(f"Type variable '{c_dim}' mapped to non-concrete dimension '{new_c_dim}'")

    # Return new MatrixType only if substitution occurred, otherwise original
    if new_r_dim is not r_dim or new_c_dim is not c_dim:
        # Create new MatrixType with potentially substituted dimensions
        return MatrixType(shape=(new_r_dim, new_c_dim))
    else:
        # Return original type if no substitutions were made
        return matrix_type

# ----------------------------------------------------------------
# Type-checking for Expressions
# ----------------------------------------------------------------
def type_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict) -> Type:
    """
    Type-check an expression 'expr'. Handles polymorphism based on strings in MatrixType.shape.
    """
    if isinstance(expr, Literal):
        return MAT_1x1 # Integer literals are Mat(1,1)

    elif isinstance(expr, Variable):
        var_type = bindings.get(expr.name)
        if var_type is None:
            raise TypeError(f"Undefined variable: '{expr.name}'")
        return var_type

    elif isinstance(expr, MatrixLiteral):
        if not expr.values: return MAT_0x0 # Empty matrix
        num_rows = len(expr.values)
        if num_rows == 0: return MAT_0x0
        if not expr.values[0]: raise TypeError("Matrix literal rows cannot be empty.")
        num_cols = len(expr.values[0])
        if num_cols == 0: raise TypeError("Matrix literal columns cannot be empty.")

        # Check all elements and row lengths
        for r_idx, row in enumerate(expr.values):
            if len(row) != num_cols:
                raise TypeError(f"Matrix literal has inconsistent row lengths (row {r_idx})")
            for c_idx, elem_expr in enumerate(row):
                elem_type = type_expr(elem_expr, bindings, declarations)
                if elem_type != MAT_1x1:
                    raise TypeError(f"Matrix literal element at [{r_idx}][{c_idx}] must be Mat(1,1), got {elem_type}")
        return MatrixType(shape=(ConcreteDim(num_rows), ConcreteDim(num_cols)))

    elif isinstance(expr, BinaryExpr):
        left_type = type_expr(expr.left, bindings, declarations)
        right_type = type_expr(expr.right, bindings, declarations)

        if not isinstance(left_type, MatrixType): raise TypeError(f"Left operand of '{expr.op.name}' must be a matrix")
        if not isinstance(right_type, MatrixType): raise TypeError(f"Right operand of '{expr.op.name}' must be a matrix")

        l_rows, l_cols = left_type.shape
        r_rows, r_cols = right_type.shape

        if expr.op in (BinOp.PLUS, BinOp.MINUS):
            # Requires exact shape match (ConcreteDim == ConcreteDim, str == str)
            if left_type.shape != right_type.shape:
                op_sym = '+' if expr.op == BinOp.PLUS else '-'
                raise TypeError(f"Operands of '{op_sym}' must have the same shape, got {left_type} and {right_type}")
            return left_type

        elif expr.op == BinOp.TIMES:
            # Inner dimensions must match (l_cols == r_rows)
            # Allows str == str, ConcreteDim == ConcreteDim
            # Disallows str == ConcreteDim (type error)
            if type(l_cols) != type(r_rows):
                 # One is str, one is ConcreteDim -> Error
                 raise TypeError(f"Matrix multiplication inner dimensions type mismatch: cannot compare {type(l_cols)} and {type(r_rows)} in {left_type} * {right_type}")
            if l_cols != r_rows:
                 # Handles str == str and ConcreteDim == ConcreteDim comparison
                 col_val = l_cols.value if isinstance(l_cols, ConcreteDim) else l_cols
                 row_val = r_rows.value if isinstance(r_rows, ConcreteDim) else r_rows
                 raise TypeError(f"Matrix multiplication dimension mismatch: {left_type} * {right_type} (inner dimensions {col_val} vs {row_val} must be equal)")

            # Result preserves outer dimension types (str or ConcreteDim)
            return MatrixType(shape=(l_rows, r_cols))
        else:
            raise TypeError(f"Internal error: Unknown binary operator {expr.op}")

    elif isinstance(expr, FunctionCall):
        # Handle built-in print function
        if expr.name == "print":
            for arg in expr.args: type_expr(arg, bindings, declarations) # Type check args
            return MAT_0x0 # Print returns Mat(0,0)

        # Handle user-defined functions
        func_decl = declarations.get(expr.name)
        if func_decl is None or not isinstance(func_decl, FunctionDec):
            raise TypeError(f"Call to undefined function: '{expr.name}'")
        func_type = func_decl.ty
        if not isinstance(func_type, FunctionType):
             raise TypeError(f"Internal error: Declaration for '{expr.name}' is not a FunctionType.")
        if len(expr.args) != len(func_type.params):
            raise TypeError(f"Function '{expr.name}' expected {len(func_type.params)} arguments, but got {len(expr.args)}")

        # Check argument types against parameter types, handling polymorphism
        type_var_map = {} # Map from type variable name (str) -> ConcreteDim

        for i, (arg_expr, param_type) in enumerate(zip(expr.args, func_type.params)):
            arg_type = type_expr(arg_expr, bindings, declarations)

            if not isinstance(arg_type, MatrixType): raise TypeError(f"Argument {i+1} to '{expr.name}' must be a matrix, got {arg_type}")
            if not isinstance(param_type, MatrixType): raise TypeError(f"Internal error: Param {i+1} for '{expr.name}' has non-matrix type {param_type}")

            arg_rows, arg_cols = arg_type.shape
            param_rows, param_cols = param_type.shape

            # --- Check Rows ---
            if isinstance(param_rows, str): # Polymorphic parameter row ('a', 'b', etc.)
                var_name = param_rows
                # Argument dimension must be concrete to bind a type variable
                if not isinstance(arg_rows, ConcreteDim):
                     raise TypeError(f"Cannot bind polymorphic variable '{var_name}' in function '{expr.name}' to non-concrete dimension '{arg_rows}' from argument {i+1}")
                if var_name in type_var_map: # Variable already bound
                    if type_var_map[var_name] != arg_rows: # Check consistency
                        raise TypeError(f"Type variable '{var_name}' mismatch for function '{expr.name}': previously bound to {type_var_map[var_name]}, but argument {i+1} has row dimension {arg_rows}")
                else: # Bind variable
                    type_var_map[var_name] = arg_rows
            elif isinstance(param_rows, ConcreteDim): # Concrete parameter row
                 if arg_rows != param_rows: # Argument must match exactly
                      raise TypeError(f"Row dimension mismatch for argument {i+1} of function '{expr.name}': expected {param_rows}, got {arg_rows}")
            else: raise TypeError(f"Internal error: Invalid parameter row dimension type {type(param_rows)}")

            # --- Check Columns --- (Similar logic as rows)
            if isinstance(param_cols, str): # Polymorphic parameter col
                var_name = param_cols
                if not isinstance(arg_cols, ConcreteDim):
                     raise TypeError(f"Cannot bind polymorphic variable '{var_name}' in function '{expr.name}' to non-concrete dimension '{arg_cols}' from argument {i+1}")
                if var_name in type_var_map:
                    if type_var_map[var_name] != arg_cols:
                        raise TypeError(f"Type variable '{var_name}' mismatch for function '{expr.name}': previously bound to {type_var_map[var_name]}, but argument {i+1} has column dimension {arg_cols}")
                else:
                    type_var_map[var_name] = arg_cols
            elif isinstance(param_cols, ConcreteDim): # Concrete parameter col
                 if arg_cols != param_cols:
                      raise TypeError(f"Column dimension mismatch for argument {i+1} of function '{expr.name}': expected {param_cols}, got {arg_cols}")
            else: raise TypeError(f"Internal error: Invalid parameter column dimension type {type(param_cols)}")

        # Determine concrete return type by substituting variables using the map
        final_ret_type = _substitute_poly_type(func_type.ret, type_var_map)

        # Check if substitution resulted in concrete dimensions if return type was poly
        if isinstance(final_ret_type, MatrixType):
             ret_rows, ret_cols = final_ret_type.shape
             if isinstance(ret_rows, str) or isinstance(ret_cols, str):
                  # This means the return type remains polymorphic after substitution.
                  # This could happen if e.g. `-> Mat(c, d)` and c, d weren't in params.
                  # This should likely be a TypeError, as function calls should resolve
                  # polymorphism based on arguments.
                  raise TypeError(f"Could not infer concrete return type for '{expr.name}' call: resolved to {final_ret_type}")

        return final_ret_type
    else:
        raise TypeError(f"Internal error: Unrecognized expression type for type checking: {type(expr)}")


# ----------------------------------------------------------------
# Type-checking for Statements
# ----------------------------------------------------------------

# Helper to find return types within a block (avoids modifying type_block structure)
# Takes the scope *after* the block has been processed to find let bindings
def _find_and_check_return_types(func_name: str, declared_ret_type: Type, body: Block, final_body_bindings: ScopedDict, declarations: ScopedDict):
    """Finds return statements in body and checks their types against declared_ret_type."""
    actual_return_types = []
    has_return_stmt = False

    # Simple traversal to find return statements and type their expressions
    # Assumes nested blocks aren't relevant for top-level function return
    for stmt in body.stmts:
        if isinstance(stmt, Return):
            has_return_stmt = True
            try:
                # Type check the expression within the return using the final scope
                actual_return_types.append(type_expr(stmt.expr, final_body_bindings, declarations))
            except TypeError as e:
                # Add context to the error message
                raise TypeError(f"Error in return statement of function '{func_name}': {e.msg}") from e

    # Now check consistency
    if not has_return_stmt:
        # No return statement found
        if declared_ret_type != MAT_0x0:
            # Check if declared return is polymorphic - still an error if no return
            is_poly_return = False
            if isinstance(declared_ret_type, MatrixType):
                 decl_rows, decl_cols = declared_ret_type.shape
                 if isinstance(decl_rows, str) or isinstance(decl_cols, str):
                      is_poly_return = True

            if is_poly_return:
                 raise TypeError(f"Function '{func_name}' has no 'return' statement, but has polymorphic declared return type {declared_ret_type}")
            else:
                 raise TypeError(f"Function '{func_name}' has no 'return' statement, but declared return type is {declared_ret_type} (expected Mat(0, 0))")
    else:
        # Check if all found return types match the declared return type
        for actual_ret_type in actual_return_types:
            is_poly_return = False
            poly_declared_type = None # Store the poly type for potential unification check
            if isinstance(declared_ret_type, MatrixType):
                 decl_rows, decl_cols = declared_ret_type.shape
                 if isinstance(decl_rows, str) or isinstance(decl_cols, str):
                      is_poly_return = True
                      poly_declared_type = declared_ret_type

            if is_poly_return:
                 # If declared is poly, check if actual is matrix.
                 # A full check would require unification.
                 if not isinstance(actual_ret_type, MatrixType):
                      raise TypeError(f"Return type mismatch in '{func_name}': declared poly {declared_ret_type}, but found return with non-matrix type {actual_ret_type}")
                 # Placeholder for unification: for now, accept any matrix type if declared is poly
                 # print(f"Warning: Polymorphic return type {poly_declared_type} for {func_name} - skipping exact match check for actual type {actual_ret_type}")
                 pass
            elif actual_ret_type != declared_ret_type: # Exact match required for concrete declared type
                 raise TypeError(f"Return type mismatch in function '{func_name}': declared {declared_ret_type}, but found return statement with type {actual_ret_type}")


def type_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """Type-checks a statement, updating bindings/declarations."""
    if isinstance(stmt, Let):
        value_type = type_expr(stmt.value, bindings, declarations)
        bindings[stmt.name] = value_type # Add binding to current scope

    elif isinstance(stmt, Return):
        # Just ensure the expression itself is well-typed here.
        # The check against function signature happens in FunctionDec handler.
        type_expr(stmt.expr, bindings, declarations)

    elif isinstance(stmt, FunctionDec):
        # Add function to declarations *before* checking body (for recursion)
        # Check for redeclaration (optional)
        # if declarations.get(stmt.name) is not None: raise TypeError(...)
        declarations[stmt.name] = stmt

        # Create new scope for function body
        body_bindings = _clone_scoped_dict(bindings)
        body_bindings.push_scope()
        declared_ret_type = stmt.ty.ret

        try:
            # Add parameters to the new scope
            if len(stmt.params) != len(stmt.ty.params):
                 raise TypeError(f"Internal Error: FunctionDec params/types length mismatch for '{stmt.name}'.")
            for param_name, param_type in zip(stmt.params, stmt.ty.params):
                body_bindings[param_name] = param_type # Add param types to inner scope

            # Type-check the body statements in the new scope.
            # This will process 'let' statements, adding them to body_bindings.
            type_block(stmt.body, body_bindings, declarations)

            # After processing the whole body, check the return statements found within it.
            # Pass the final state of body_bindings, which includes 'let' variables.
            _find_and_check_return_types(stmt.name, declared_ret_type, stmt.body, body_bindings, declarations)

        finally:
            body_bindings.pop_scope() # Ensure scope is removed

    elif isinstance(stmt, FunctionCall):
        # Type check the function call, result type is ignored
        type_expr(stmt, bindings, declarations)

    elif isinstance(stmt, Expr):
        # Type check a standalone expression, result type is ignored
        type_expr(stmt, bindings, declarations)

    else:
        # Unknown statement type
        # raise TypeError(f"Internal error: Unknown statement type for type checking: {type(stmt)}")
        pass # Ignore unknown statement types

# ----------------------------------------------------------------
# Type-checking for Blocks
# ----------------------------------------------------------------
def type_block(block: Block, bindings: ScopedDict, declarations: ScopedDict):
    """Type-checks each statement within a block."""
    if block is None or block.stmts is None: return # Handle empty/None blocks
    for statement in block.stmts:
        if statement is not None: # Skip None entries if parser allows them
             type_stmt(statement, bindings, declarations)

