# parse.py

from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import Node
from parsimonious.grammar import Grammar
from pathlib import Path
import sys
import re

# --- Explicit Imports ---
# Import only the necessary classes to avoid wildcard issues
from AST import (
    Statement, Expr, Block, BinaryExpr, Let, Literal, Variable,
    MatrixLiteral, FunctionCall, FunctionDec, Return, BinOp
    # Note: Print, VectorLiteral, ExpressionStatement might be unused now
)
from typ import (
    Type, Dim, ConcreteDim, TypeVarDim, MatrixType, FunctionType
)
# --- End Explicit Imports ---


# ────────────────────────────────────────────────────────────────────────────
# Helper functions for expression handling
# ────────────────────────────────────────────────────────────────────────────
def _expr(item):
    """Return an Expr; wrap bare identifier strings as Variable."""
    if isinstance(item, Expr):
        return item
    if isinstance(item, str):
        # Avoid wrapping operators if they somehow leak through
        if item in ['+', '-', '*']:
             return item
        return Variable(item)
    # Handle lists potentially returned by visitor methods
    if isinstance(item, list) and item:
        if len(item) == 1:
             # Recurse on the single item within the list
             return _expr(item[0])
    if isinstance(item, int):
        return Literal(item)
    return item # Return item as is if none of the above match

def _first_expr(obj):
    """Return the first Expr/value inside obj, unwrapping one‑element lists."""
    # Modified: Do not wrap int in Literal here.
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, list) and obj:
        # If list contains a single item, recurse
        if len(obj) == 1:
             return _first_expr(obj[0])
    # Return int or str directly if not already Expr
    if isinstance(obj, (int, str)):
        return obj
    return obj # Return obj as is if none of the above match

# ────────────────────────────────────────────────────────────────────────────
# Main Visitor Class
# ────────────────────────────────────────────────────────────────────────────
class MatrixVisitor(NodeVisitor):
    """Visitor for the parse tree that converts a MatLang parse tree (as defined in grammar.peg)
    into an Abstract Syntax Tree (AST). Each visit method corresponds to a grammar rule.
    """

    def generic_visit(self, node, visited_children):
        """Default visit method. Returns visited children or node text."""
        return visited_children or node.text

    def visit_name(self, node, visited_children):
        """Extracts the text for a name node, stripping whitespace."""
        return node.text.strip()

    # ----------------------------------------------------------------
    # Program: program = (statement / comment / emptyline)*
    # ----------------------------------------------------------------
    def visit_program(self, node, visited_children):
        """Processes the top-level program rule."""
        statements = []
        # Filter out None results from comments/empty lines and flatten lists
        for child_group in visited_children:
             items_to_process = child_group if isinstance(child_group, list) else [child_group]
             for item in items_to_process:
                  if isinstance(item, Statement):
                       statements.append(item)
                  # Handle potential deeper nesting if a visit method returns [Stmt]
                  elif isinstance(item, list) and item and isinstance(item[0], Statement):
                       statements.append(item[0])
                  # Ignore None or other non-statement results (like comments)
        return Block(statements)

    # ----------------------------------------------------------------
    # Let Statement: let_stmt = "let" ws name ws "=" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_let_stmt(self, node, ch):
        """Processes a let statement."""
        # Indices: let[0] ws[1] name[2] ws[3] =[4] ws[5] expr[6] ;[7] ws[8]
        var_name = ch[2] # Result of visit_name
        expr_node = _expr(ch[6]) # Result of visit_expr
        return Let(var_name, expr_node)

    # ----------------------------------------------------------------
    # Expression Statement: expr_stmt = expr ";" ws
    # ----------------------------------------------------------------
    def visit_expr_stmt(self, node, ch):
        """Processes an expression statement."""
        # Indices: expr[0] ;[1] ws[2]
        # Return the raw expression node to match expected AST
        return _expr(ch[0])

    # ----------------------------------------------------------------
    # Function Definition
    # func_def = "def" ws name ws "(" ws params? ws ")" ws ("->" ws type)? ws "{" ws func_body "}" ws
    # ----------------------------------------------------------------
    def visit_func_def(self, node, visited_children):
        """Processes a function definition."""
        # Indices (approximate):
        # "def" ws name ws "(" ws params? ws ")" ws ("->" ws type)? ws "{" ws func_body "}" ws
        #  0    1   2   3   4   5    6     7   8   9       10        11 12 13     14    15 16
        func_name = visited_children[2]

        param_info_list = [] # Default: no parameters
        params_node_result = visited_children[6]
        # Check if params node exists and contains the list from visit_params
        if isinstance(params_node_result, list) and len(params_node_result) > 0:
            potential_params = params_node_result[0]
            if isinstance(potential_params, list) and all(isinstance(p, tuple) for p in potential_params):
                 param_info_list = potential_params

        # Convert lists to tuples to match expected AST structure
        param_names = tuple([name for name, type in param_info_list])
        param_types = tuple([type for name, type in param_info_list])

        # Handle optional return type
        return_type = MatrixType((ConcreteDim(0), ConcreteDim(0))) # Default type
        return_type_group_result = visited_children[10]

        # Check if the optional group matched and find the Type object
        if isinstance(return_type_group_result, list) and len(return_type_group_result) == 1 and isinstance(return_type_group_result[0], list):
            inner_list = return_type_group_result[0]
            found_type_node = None
            for item in inner_list:
                 if isinstance(item, Type):
                      found_type_node = item
                      break
            if found_type_node:
                return_type = found_type_node
            else: # Should not happen if grammar/visitor for type is correct
                raise Exception(f"visit_func_def: Optional return type group's inner list did not contain a Type object: {inner_list}")

        # Find the function body Block node
        func_body_node = visited_children[14]
        if isinstance(func_body_node, Block):
            func_body = func_body_node
        else: # Default to empty block if parsing/visiting body yielded something else
             func_body = Block(stmts=[])

        # Create FunctionType and FunctionDec nodes using tuples for params
        func_type = FunctionType(params=param_types, ret=return_type)
        return FunctionDec(name=func_name, params=param_names, body=func_body, ty=func_type)

    # ----------------------------------------------------------------
    # Parameters List: params = param ("," ws param)*
    # ----------------------------------------------------------------
    def visit_params(self, node, visited_children):
        """Processes function parameter list. Returns a list of (name, type) tuples."""
        # visited_children structure: [param_result, list_of_groups]
        first_param_result = visited_children[0] # Result from visit_param
        # Validate structure
        if not (isinstance(first_param_result, tuple) and len(first_param_result) == 2):
             raise Exception(f"visit_params: Expected tuple for first_param_result, got {type(first_param_result)}: {first_param_result}")

        rest_params = []
        # visited_children[1] is list of groups like [",", ws, param_result]
        for group in visited_children[1]:
            param_result = group[2] # Get the param result from the group
            # Validate structure
            if not (isinstance(param_result, tuple) and len(param_result) == 2):
                 raise Exception(f"visit_params: Expected tuple for subsequent param_result, got {type(param_result)}: {param_result}")
            rest_params.append(param_result)

        all_params = [first_param_result] + rest_params
        return all_params # Returns list of (name, type) tuples

    # ----------------------------------------------------------------
    # Parameter: param = name ws ":" ws type
    # ----------------------------------------------------------------
    def visit_param(self, node, visited_children):
        """Processes single parameter with type. Returns (name: str, type: Type)"""
        # Indices: name[0] ws[1] :[2] ws[3] type[4]
        param_name = visited_children[0] # Result of visit_name
        param_type = visited_children[4] # Result of visit_type
        # Validate types
        if not isinstance(param_name, str):
            raise Exception(f"visit_param: Expected string for parameter name, got {type(param_name)}: {param_name}")
        if not isinstance(param_type, Type):
            type_node_text = node.children[4].text # Get original text for error msg
            raise Exception(f"visit_param: Expected Type for parameter type (for '{param_name}'), got {type(param_type)}: {param_type}. Parsed type text was: '{type_node_text}'")
        result = (param_name, param_type)
        return result

    # ----------------------------------------------------------------
    # Type: type = "Mat(" ws dim ws "," ws dim ws ")" ws
    # ----------------------------------------------------------------
    def visit_type(self, node, visited_children):
        """Processes a type annotation."""
        # Helper to convert dim result (int/str) to ConcreteDim or str (for pickle compatibility)
        def extract_dim_for_pickle(val):
            processed_val = _first_expr(val) # Use helper to unwrap potential lists/nodes
            if isinstance(processed_val, int):
                return ConcreteDim(processed_val) # Concrete dimensions use the class
            if isinstance(processed_val, str):
                if processed_val.isalpha():
                     return processed_val # Polymorphic dimensions are strings
                elif processed_val.isdigit():
                     return ConcreteDim(int(processed_val)) # Handle numbers parsed as name
                else:
                     raise ValueError(f"Invalid dimension string value: {processed_val}")
            raise TypeError(f"Unexpected type/value for dimension: {type(processed_val)}: {processed_val!r}")

        # Indices: Mat([0] ws[1] dim[2] ws[3] ,[4] ws[5] dim[6] ws[7] )[8] ws[9]
        rows = extract_dim_for_pickle(visited_children[2]) # Result of visit_dim
        cols = extract_dim_for_pickle(visited_children[6]) # Result of visit_dim
        return MatrixType(shape=(rows, cols))

    # ----------------------------------------------------------------
    # Dim: dim = name / number
    # ----------------------------------------------------------------
    def visit_dim(self, node, visited_children):
        """Processes a dimension (name or number)."""
        # visited_children[0] is result of visit_name (str) or visit_number (int)
        return visited_children[0]

    # ----------------------------------------------------------------
    # Comments / Empty Lines
    # ----------------------------------------------------------------
    def visit_comment(self, node, visited_children):
        """Ignores comments."""
        return None
    def visit_emptyline(self, node, visited_children):
        """Ignores empty lines."""
        return None

    # ----------------------------------------------------------------
    # Function Body: func_body = (statement / comment / emptyline)*
    # ----------------------------------------------------------------
    def visit_func_body(self, node, visited_children):
        """Processes the statements within a function body."""
        statements = []
        for item in visited_children:
            # Filter results, similar to visit_program
            items_to_process = item if isinstance(item, list) else [item]
            for sub_item in items_to_process:
                 if isinstance(sub_item, Statement):
                      statements.append(sub_item)
                 elif isinstance(sub_item, list) and sub_item and isinstance(sub_item[0], Statement):
                      statements.append(sub_item[0])
        return Block(statements)

    # ----------------------------------------------------------------
    # Return Statement: return_stmt = "return" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_return_stmt(self, node, visited_children):
        """Processes a return statement."""
        # Indices: return[0] ws[1] expr[2] ;[3] ws[4]
        expr = _expr(visited_children[2])
        return Return(expr)

    # ----------------------------------------------------------------
    # Addition/Subtraction: add = mul (("+" / "-") ws mul)*
    # ----------------------------------------------------------------
    def visit_add(self, node, visited_children):
        """Processes addition/subtraction, maintaining left-associativity."""
        # visited_children = [mul_result, list_of_groups]
        # group = [op_result, ws_result, mul_result]
        left = _expr(visited_children[0])
        for group in visited_children[1]:
            op_result = group[0] # Should be '+' or '-' string from generic_visit
            right = _expr(group[2])
            op_token = None
            if isinstance(op_result, str): op_token = op_result.strip()
            elif isinstance(op_result, list) and len(op_result)==1 and isinstance(op_result[0], str): op_token = op_result[0].strip()
            else: raise Exception(f"Could not determine operator token from visit result: {op_result}")

            if op_token == "+": op_enum = BinOp.PLUS
            elif op_token == "-": op_enum = BinOp.MINUS
            else: raise Exception(f"Unknown operator token in add: '{op_token}'")
            left = _expr(left) # Ensure left is Expr node before making BinaryExpr
            left = BinaryExpr(left, right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Multiplication: mul = atom (("*") ws atom)*
    # ----------------------------------------------------------------
    def visit_mul(self, node, visited_children):
        """Processes multiplication, maintaining left-associativity."""
        # visited_children = [atom_result, list_of_groups]
        # group = [op_result, ws_result, atom_result]
        left = _expr(visited_children[0])
        for group in visited_children[1]:
            op_result = group[0] # Should be '*' string
            right = _expr(group[2])
            op_token = None
            if isinstance(op_result, str): op_token = op_result.strip()
            elif isinstance(op_result, list) and len(op_result)==1 and isinstance(op_result[0], str): op_token = op_result[0].strip()
            else: raise Exception(f"Could not determine operator token from visit result: {op_result}")

            if op_token == "*": op_enum = BinOp.TIMES
            else: raise Exception(f"Unknown operator token in mul: '{op_token}'")
            left = _expr(left) # Ensure left is Expr node
            left = BinaryExpr(left, right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Atom: atom = matrix / vector / number_literal / func_call / var
    # ----------------------------------------------------------------
    def visit_atom(self, node, visited_children):
        """Processes an atomic expression."""
        # Result is already processed by one of the child rules (matrix, vector, etc.)
        return _expr(visited_children[0])

    # ----------------------------------------------------------------
    # Variable: var = name
    # ----------------------------------------------------------------
    def visit_var(self, node, visited_children):
        """Processes a variable reference."""
        # visited_children[0] is the result of visit_name (a string)
        return Variable(visited_children[0])

    # ----------------------------------------------------------------
    # Number Literal: number_literal = number
    # ----------------------------------------------------------------
    def visit_number_literal(self, node, visited_children):
        """Processes a number literal."""
        # visited_children[0] should be the int from visit_number
        num_val = visited_children[0]
        if isinstance(num_val, int):
            return Literal(num_val)
        # Fallback just in case
        try: return Literal(int(node.text.strip()))
        except ValueError: raise Exception(f"Could not parse number literal: {node.text}")

    def visit_number(self, node, visited_children):
        """Processes the number token rule."""
        # number = ~r"[0-9]+" ws
        try: return int(node.text.strip())
        except ValueError: raise Exception(f"Could not parse number: {node.text}")

    # ----------------------------------------------------------------
    # Function Call: func_call = name ws "(" ws arguments? ws ")" ws
    # ----------------------------------------------------------------
    def visit_func_call(self, node, visited_children):
        """Processes a function call."""
        # Indices: name[0] ws[1] ([2] ws[3] arguments?[4] ws[5] )[6] ws[7]
        func_name = visited_children[0] # Result of visit_name
        args = []
        # arguments? returns list containing result of visit_arguments if matched
        args_node_result = visited_children[4]
        if isinstance(args_node_result, list) and len(args_node_result) > 0:
             potential_args = args_node_result[0] # Result often wrapped by '?'
             if isinstance(potential_args, list):
                  args = [_expr(arg) for arg in potential_args] # Ensure all are Expr
        return FunctionCall(func_name, args)

    # ----------------------------------------------------------------
    # Arguments: arguments = expr ("," ws expr)*
    # ----------------------------------------------------------------
    def visit_arguments(self, node, visited_children):
        """Processes a list of function arguments."""
        # Indices: expr[0] list_of_groups[1] where group is [",", ws, expr]
        first = _expr(visited_children[0])
        rest = [_expr(g[2]) for g in visited_children[1]]
        return [first] + rest # Return a flat list of Expr nodes

    # ----------------------------------------------------------------
    # Matrix Literal: matrix = "[" ws vector ("," ws vector)* "]" ws
    # ----------------------------------------------------------------
    def visit_matrix(self, node, visited_children):
        """Processes a matrix literal."""
        # Indices: [ [0] ws[1] vector[2] list_of_groups[3] ] [4] ws[5]
        first_vec_result = visited_children[2] # Result of visit_vector (MatrixLiteral)
        rest_vec_results = [g[2] for g in visited_children[3]] # g = [",", ws, vector_result]
        vector_results = [first_vec_result] + rest_vec_results

        row_lists = []
        # Each item 'v' in vector_results is a MatrixLiteral from visit_vector
        for v in vector_results:
            if isinstance(v, MatrixLiteral) and len(v.values) == 1:
                # Extract the single row (list of elements) from the MatrixLiteral
                row_lists.append(v.values[0])
            else:
                # Raise error if it's not the expected single-row MatrixLiteral
                raise Exception(f"Invalid matrix row structure: Expected single-row MatrixLiteral, got {type(v)}: {repr(v)}")
        # node_type might be needed if MatrixLiteral.__eq__ uses it
        return MatrixLiteral(values=row_lists, node_type='MatrixLiteral')


    # ----------------------------------------------------------------
    # Vector Literal: vector = "[" ws expr ("," ws expr)* "]" ws
    # ----------------------------------------------------------------
    def visit_vector(self, node, visited_children):
        """Processes a vector literal, returning it as a single-row MatrixLiteral."""
        # Indices: [ [0] ws[1] expr[2] list_of_groups[3] ] [4] ws[5]
        first_expr = _expr(visited_children[2])
        rest_exprs = [_expr(g[2]) for g in visited_children[3]] # g is [",", ws, expr]
        elements = [first_expr] + rest_exprs
        # node_type might be needed if MatrixLiteral.__eq__ uses it
        return MatrixLiteral(values=[elements], node_type='MatrixLiteral')

    # ----------------------------------------------------------------
    # Print Statement: print_stmt = "print" ws "(" ws expr ("," ws expr)* ws ")" ws ";" ws
    # ----------------------------------------------------------------
    def visit_print_stmt(self, node, ch):
        """Processes a print statement, returning a FunctionCall node."""
        # Indices: print[0] ws[1] ([2] ws[3] expr[4] list_of_groups[5] ws[6] )[7] ws[8] ;[9] ws[10]
        first_arg = _expr(ch[4])
        rest_args = [_expr(g[2]) for g in ch[5]] # g is [",", ws, expr]
        all_args = [first_arg] + rest_args
        return FunctionCall(name="print", args=all_args)


# ----------------------------------------------------------------
# parse() function
# ----------------------------------------------------------------
def parse(file_name: str):
    """Parse a MatLang file and return the AST"""
    grammar_path = Path("grammar.peg")
    if not grammar_path.is_file():
         raise FileNotFoundError("grammar.peg not found in the current directory.")
    grammar = Grammar(grammar_path.read_text())

    source_path = Path(file_name)
    if not source_path.is_file():
         raise FileNotFoundError(f"Input file not found: {file_name}")
    source_text = source_path.read_text()

    try:
        tree = grammar.parse(source_text)
    except Exception as parse_error:
         print(f"\n!!! PARSING ERROR in {file_name} !!!")
         print(parse_error)
         raise parse_error

    visitor = MatrixVisitor()
    try:
        ast = visitor.visit(tree)
    except Exception as visit_error:
         # Add more context to visitor errors
         print(f"\n!!! VISITOR ERROR processing {file_name} !!!")
         print(f"Error Type: {type(visit_error).__name__}")
         print(f"Error Args: {visit_error.args}")
         # Attempt to get node info if VisitationError
         if hasattr(visit_error, 'original_exception') and hasattr(visit_error, 'node'):
              print(f"Occurred near node: {visit_error.node.expr_name}")
              print(f"Node text: '{visit_error.node.text}'")
         elif hasattr(visit_error, '__traceback__'): # General exception traceback
              tb = visit_error.__traceback__
              while tb.tb_next: # Find innermost frame in visitor
                  if 'visit_' in tb.tb_frame.f_code.co_name:
                       break
                  tb = tb.tb_next
              print(f"Occurred around visitor method: {tb.tb_frame.f_code.co_name}, line: {tb.tb_lineno}")

         raise visit_error # Re-raise after printing info

    return ast

# ----------------------------------------------------------------
# Command-Line Entry Point
# ----------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse.py [input_file]")
        exit(1)
    input_file = sys.argv[1]
    print(f"Parsing {input_file}...")
    try:
        print("\n--- Building AST ---")
        ast = parse(input_file) # Call the parse function directly
        print("\n--- Generated AST ---")
        print(repr(ast))
        print("\nParsing and AST construction successful.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"\nAn error occurred during parsing or visiting:")
        import traceback
        traceback.print_exc()

