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
        return f"ConcreteDim(value={self.value!r})"

    def __str__(self):
        return str(self.value)

@dataclass(eq=True, frozen=True) # Frozen makes it hashable, usable as dict key
class TypeVarDim(Dim):
    """Represents a type variable for dimensions."""
    name: str

    def __repr__(self):
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
    params: tuple[Type, ...] # Use tuple type hint
    ret: Type

    def __repr__(self):
        return f"FunctionType(params={self.params!r}, ret={self.ret!r})"

    def __str__(self):
        param_strs = [str(p) for p in self.params]
        return f"({', '.join(param_strs)}) -> {self.ret}"

# ----------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------
def _clone_scoped_dict(src: ScopedDict) -> ScopedDict:
    """Deep copy a ScopedDict."""
    new = ScopedDict()
    new.dicts = [d.copy() for d in src.dicts] # Create copy of list of dicts
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
        if not isinstance(new_r_dim, ConcreteDim):
             raise TypeError(f"Type variable '{r_dim}' mapped to non-concrete dimension '{new_r_dim}'")

    # Substitute column dimension if it's a string variable found in the map
    if isinstance(c_dim, str) and c_dim in type_var_map:
        new_c_dim = type_var_map[c_dim]
        if not isinstance(new_c_dim, ConcreteDim):
             raise TypeError(f"Type variable '{c_dim}' mapped to non-concrete dimension '{new_c_dim}'")

    # Return new MatrixType only if substitution occurred, otherwise original
    if new_r_dim is not r_dim or new_c_dim is not c_dim:
        return MatrixType(shape=(new_r_dim, new_c_dim))
    else:
        return matrix_type

# ----------------------------------------------------------------
# Type-checking for Expressions
# ----------------------------------------------------------------
def type_expr(expr: Expr, bindings: ScopedDict, declarations: ScopedDict) -> Type:
    """
    Type-check an expression 'expr'.

    Parameters:
      expr          - the AST node representing the expression.
      bindings      - a dictionary mapping variable names to their types.
      declarations  - a dictionary mapping function names to their declarations (FunctionDec nodes).

    Returns:
      The type (MatrixType or FunctionType) of the expression.

    Raises:
      TypeError if the expression is ill-typed.
    """
    if isinstance(expr, Literal):
        # Integer literals are Mat(1,1)
        return MAT_1x1

    elif isinstance(expr, Variable):
        # Look up variable type in bindings
        var_type = bindings.get(expr.name) # Use get to avoid KeyError here
        if var_type is None:
            raise TypeError(f"Undefined variable: '{expr.name}'")
        return var_type

    elif isinstance(expr, MatrixLiteral):
        # Handle empty matrix literal
        if not expr.values:
            return MAT_0x0

        num_rows = len(expr.values)
        if num_rows == 0: # Should be caught by `if not expr.values` but check again
             return MAT_0x0

        # Check first row to determine expected column count
        if not expr.values[0]: # Empty first row -> Mat(N, 0) - Allowed? Spec implies Mat(0,0) is main empty type.
             # Let's assume non-ragged, so if first is empty, all should be.
             # Treat as Mat(N, 0) or raise error? Let's require Mat(0,0) for empty.
             # If we allow Mat(N, 0), need to check all rows are empty.
             # For simplicity matching spec, let's require elements.
             raise TypeError("Matrix literal rows cannot be empty unless matrix itself is.")

        num_cols = len(expr.values[0])
        if num_cols == 0: # Mat(N, 0) case
             raise TypeError("Matrix literal columns cannot be empty unless matrix itself is.")


        # Check all elements and row lengths
        for r_idx, row in enumerate(expr.values):
            if len(row) != num_cols:
                raise TypeError(f"Matrix literal has inconsistent row lengths (row {r_idx} has {len(row)}, expected {num_cols})")
            for c_idx, elem_expr in enumerate(row):
                elem_type = type_expr(elem_expr, bindings, declarations)
                # Spec says integer matrices, implies elements must be Mat(1,1)
                if elem_type != MAT_1x1:
                    raise TypeError(f"Matrix literal element at [{r_idx}][{c_idx}] must be an integer (Mat(1,1)), but got {elem_type}")

        return MatrixType(shape=(ConcreteDim(num_rows), ConcreteDim(num_cols)))

    elif isinstance(expr, BinaryExpr):
        left_type = type_expr(expr.left, bindings, declarations)
        right_type = type_expr(expr.right, bindings, declarations)

        if not isinstance(left_type, MatrixType):
            raise TypeError(f"Left operand of '{expr.op.name}' must be a matrix, but got {left_type}")
        if not isinstance(right_type, MatrixType):
            raise TypeError(f"Right operand of '{expr.op.name}' must be a matrix, but got {right_type}")

        l_rows, l_cols = left_type.shape
        r_rows, r_cols = right_type.shape

        if expr.op in (BinOp.PLUS, BinOp.MINUS):
            # Dimensions must match exactly for + and -
            # Note: This check implicitly handles ConcreteDim vs ConcreteDim.
            # It does NOT handle polymorphism across '+' or '-'. Spec doesn't mention this.
            if left_type.shape != right_type.shape:
                op_sym = '+' if expr.op == BinOp.PLUS else '-'
                raise TypeError(f"Operands of '{op_sym}' must have the same shape, but got {left_type} and {right_type}")
            # Result type is the same as operands
            return left_type

        elif expr.op == BinOp.TIMES:
            # Inner dimensions must match for * (l_cols == r_rows)
            # Need to handle comparison between ConcreteDim and potentially string type vars
            # For multiplication, type variables usually aren't involved *within* one side's shape,
            # but across the l_cols/r_rows comparison. However, our shape tuple stores str/ConcreteDim.
            # Let's assume for now that operands to '*' must have concrete inner dimensions.
            # Polymorphism is handled at the function call level primarily.
            if not isinstance(l_cols, ConcreteDim):
                 raise TypeError(f"Left operand of '*' must have concrete columns, but got '{l_cols}' in {left_type}")
            if not isinstance(r_rows, ConcreteDim):
                 raise TypeError(f"Right operand of '*' must have concrete rows, but got '{r_rows}' in {right_type}")

            if l_cols.value != r_rows.value:
                raise TypeError(f"Matrix multiplication dimension mismatch: {left_type} * {right_type} (inner dimensions {l_cols.value} and {r_rows.value} must be equal)")

            # Result type is Mat(l_rows, r_cols)
            # Need to preserve potential string type vars in outer dimensions
            return MatrixType(shape=(l_rows, r_cols))

        else:
            # Should not happen if AST construction is correct
            raise TypeError(f"Internal error: Unknown binary operator {expr.op}")

    elif isinstance(expr, FunctionCall):
        # Handle built-in print function
        if expr.name == "print":
            # Type check all arguments (any type allowed for print)
            for arg in expr.args:
                type_expr(arg, bindings, declarations)
            # Print returns Mat(0,0) according to spec
            return MAT_0x0

        # Handle user-defined functions
        func_decl = declarations.get(expr.name)
        if func_decl is None or not isinstance(func_decl, FunctionDec):
            raise TypeError(f"Call to undefined function: '{expr.name}'")

        func_type = func_decl.ty
        if not isinstance(func_type, FunctionType):
             # Should not happen if parser/declarations are correct
             raise TypeError(f"Internal error: Declaration for '{expr.name}' is not a FunctionType.")

        # Check argument count
        if len(expr.args) != len(func_type.params):
            raise TypeError(f"Function '{expr.name}' expected {len(func_type.params)} arguments, but got {len(expr.args)}")

        # Check argument types against parameter types, handling polymorphism
        type_var_map = {} # Map from type variable name (str) -> ConcreteDim

        for i, (arg_expr, param_type) in enumerate(zip(expr.args, func_type.params)):
            arg_type = type_expr(arg_expr, bindings, declarations)

            if not isinstance(arg_type, MatrixType):
                 raise TypeError(f"Argument {i+1} to function '{expr.name}' must be a matrix, but got {arg_type}")
            if not isinstance(param_type, MatrixType):
                 raise TypeError(f"Internal error: Parameter {i+1} for function '{expr.name}' has non-matrix type {param_type}")

            arg_rows, arg_cols = arg_type.shape
            param_rows, param_cols = param_type.shape

            # --- Check Rows ---
            if isinstance(param_rows, str): # Polymorphic parameter row
                var_name = param_rows
                if not isinstance(arg_rows, ConcreteDim):
                     raise TypeError(f"Cannot bind polymorphic variable '{var_name}' in function '{expr.name}' to non-concrete dimension '{arg_rows}' from argument {i+1}")
                if var_name in type_var_map:
                    # Variable already bound, check consistency
                    if type_var_map[var_name] != arg_rows:
                        raise TypeError(f"Type variable '{var_name}' mismatch for function '{expr.name}': previously bound to {type_var_map[var_name]}, but argument {i+1} has row dimension {arg_rows}")
                else:
                    # Bind variable
                    type_var_map[var_name] = arg_rows
            elif isinstance(param_rows, ConcreteDim): # Concrete parameter row
                 if arg_rows != param_rows:
                      raise TypeError(f"Row dimension mismatch for argument {i+1} of function '{expr.name}': expected {param_rows}, got {arg_rows}")
            else: # Should not happen if parser is correct
                 raise TypeError(f"Internal error: Invalid parameter row dimension type {type(param_rows)}")

            # --- Check Columns ---
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
            else: # Should not happen
                 raise TypeError(f"Internal error: Invalid parameter column dimension type {type(param_cols)}")

        # Determine concrete return type by substituting variables
        final_ret_type = _substitute_poly_type(func_type.ret, type_var_map)

        # Check if substitution resulted in concrete dimensions if needed
        if isinstance(final_ret_type, MatrixType):
             ret_rows, ret_cols = final_ret_type.shape
             if isinstance(ret_rows, str) or isinstance(ret_cols, str):
                  # This implies the return type itself was polymorphic and not fully resolved
                  # This might be okay depending on context, or an error if concrete needed
                  # For now, allow polymorphic return types if substitution didn't make them concrete
                  pass
                  # If concrete required: raise TypeError(f"Could not infer concrete return type for '{expr.name}': got {final_ret_type}")

        return final_ret_type

    else:
        raise TypeError(f"Internal error: Unrecognized expression type for type checking: {type(expr)}")


# ----------------------------------------------------------------
# Type-checking for Statements
# ----------------------------------------------------------------

# Helper to find return types within a block (avoids modifying type_block)
def _find_return_types(block: Block, bindings: ScopedDict, declarations: ScopedDict) -> list[Type]:
    """Recursively finds the types of all Return statements within a block."""
    return_types = []
    for stmt in block.stmts:
        if isinstance(stmt, Return):
            try:
                return_types.append(type_expr(stmt.expr, bindings, declarations))
            except TypeError as e:
                # Propagate type errors found within the return expression
                raise e
        elif isinstance(stmt, FunctionDec):
             # Don't recurse into nested function definitions for *this* function's return check
             pass
        elif isinstance(stmt, Block): # Should not happen directly in stmts list, but for safety
             return_types.extend(_find_return_types(stmt, bindings, declarations))
        # Add checks for other block-like structures if they exist (e.g., If, While)
    return return_types


def type_stmt(stmt: Statement, bindings: ScopedDict, declarations: ScopedDict):
    """
    Type-check a statement. Updates bindings and declarations.

    Parameters:
      stmt          - the AST node representing the statement.
      bindings      - a dictionary mapping variable names to their types (modified for Let).
      declarations  - a dictionary mapping function names to their declarations (modified for FunctionDec).

    Raises:
      TypeError if the statement is ill-typed.
    """
    if isinstance(stmt, Let):
        # Type check the expression being assigned
        value_type = type_expr(stmt.value, bindings, declarations)
        # Add the variable and its type to the current scope
        bindings[stmt.name] = value_type

    elif isinstance(stmt, Return):
        # Type check the expression being returned.
        # The check against the function's declared type happens in the FunctionDec handler.
        type_expr(stmt.expr, bindings, declarations)

    elif isinstance(stmt, FunctionDec):
        # 1. Check for redeclaration in the current scope (optional, but good practice)
        if declarations.get(stmt.name) is not None:
             # Allow redeclaration for simplicity, or raise error:
             # raise TypeError(f"Function '{stmt.name}' already declared.")
             pass

        # 2. Add function declaration to the declarations dict *before* checking body
        #    This allows for recursive function calls.
        declarations[stmt.name] = stmt

        # 3. Create a new scope for the function body
        body_bindings = _clone_scoped_dict(bindings)
        body_bindings.push_scope()

        # 4. Add parameters to the function's scope
        if len(stmt.params) != len(stmt.ty.params):
             raise TypeError(f"Internal Error: FunctionDec AST params list length ({len(stmt.params)}) doesn't match FunctionType params length ({len(stmt.ty.params)}) for '{stmt.name}'.")
        for param_name, param_type in zip(stmt.params, stmt.ty.params):
            body_bindings[param_name] = param_type

        # 5. Type check the function body statements recursively
        #    We need to check the types of any 'return' statements inside.
        try:
            # Find all return types within the body
            actual_return_types = _find_return_types(stmt.body, body_bindings, declarations)

            declared_ret_type = stmt.ty.ret

            if not actual_return_types:
                # No return statement found
                # Check if declared return type is Mat(0,0)
                if declared_ret_type != MAT_0x0:
                    # Allow if declared return is polymorphic? No, spec implies concrete check.
                    raise TypeError(f"Function '{stmt.name}' has no 'return' statement, but declared return type is {declared_ret_type} (expected Mat(0, 0))")
            else:
                # Check if all found return types match the declared return type
                # This needs to handle potential polymorphism in the declared type.
                # We don't substitute here, as the declared type might be polymorphic intentionally.
                # Instead, we check if each actual return type *could* match the declared one.
                # This basic checker assumes concrete return types must match exactly.
                # A more advanced checker would use unification for polymorphism.
                for actual_ret_type in actual_return_types:
                     # Simple check: if declared is concrete, actual must match.
                     # If declared is polymorphic (contains strings), this check is insufficient.
                     # Let's assume for now declared return types in definitions must be concrete
                     # unless we implement full polymorphism checks.
                     # Revisit based on how polymorphism bonus is handled.
                     if isinstance(declared_ret_type, MatrixType):
                          decl_rows, decl_cols = declared_ret_type.shape
                          if isinstance(decl_rows, str) or isinstance(decl_cols, str):
                               # Declared return is polymorphic - skip exact check for now
                               # A real implementation would need unification here.
                               pass
                          elif actual_ret_type != declared_ret_type:
                               raise TypeError(f"Return type mismatch in function '{stmt.name}': declared {declared_ret_type}, but found return statement with type {actual_ret_type}")
                     elif actual_ret_type != declared_ret_type:
                           raise TypeError(f"Return type mismatch in function '{stmt.name}': declared {declared_ret_type}, but found return statement with type {actual_ret_type}")

            # Also type check non-return statements in the body for other errors
            type_block(stmt.body, body_bindings, declarations)

        finally:
            # Ensure scope is popped even if errors occur
            body_bindings.pop_scope()

    elif isinstance(stmt, FunctionCall):
        # Type check the function call expression, discard the resulting type
        type_expr(stmt, bindings, declarations)

    elif isinstance(stmt, Expr):
        # Type check a standalone expression used as a statement
        type_expr(stmt, bindings, declarations)

    # Add cases for other statement types if they exist (e.g., Print, ExpressionStatement)
    # Note: The parser was changed to make print a FunctionCall and expr; just an Expr.
    # If you had other statement types like IfStmt, WhileStmt, add them here.

    else:
        # Ignore unknown nodes or raise error? Let's ignore for now.
        # raise TypeError(f"Internal error: Unknown statement type for type checking: {type(stmt)}")
        pass


# ----------------------------------------------------------------
# Type-checking for Blocks
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
    # Check if block or stmts is None before iterating
    if block is None or block.stmts is None:
        # Handle case where a block might be empty or None (e.g., empty function body parsed as None)
        return # Nothing to check

    # Iterate over each statement in the block and type-check it.
    for statement in block.stmts:
        if statement is not None: # Skip None entries if parser allows them
             type_stmt(statement, bindings, declarations)

