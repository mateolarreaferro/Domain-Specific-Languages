# parse.py

from parsimonious.nodes import NodeVisitor  # Allows walking the parse tree.
from parsimonious.nodes import Node         # Represents nodes in the parse tree.
from parsimonious.grammar import Grammar    # Loads and works with the PEG grammar.
from pathlib import Path                    # Helps read files (grammar and input).
from AST import * # Import AST classes (Block, Let, etc.)
from typ import * # Import type-checking utilities
import sys                                  # Provides command-line argument support
import re                                   # Regular expressions (if needed)

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
    """Return the first Expr inside obj, unwrapping one‑element lists."""
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, list) and obj:
        return _first_expr(obj[0])
    if isinstance(obj, int):
        return Literal(obj)
    return obj

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
        return node.text.strip()

    # ----------------------------------------------------------------
    # Program
    # ----------------------------------------------------------------
    def visit_program(self, node, visited_children):
        statements = []
        for child_group in visited_children:
             items_to_process = child_group if isinstance(child_group, list) else [child_group]
             for item in items_to_process:
                  if isinstance(item, Statement):
                       statements.append(item)
                  elif isinstance(item, list) and item and isinstance(item[0], Statement):
                       statements.append(item[0])
        return Block(statements)

    # ----------------------------------------------------------------
    # Let Statement: let_stmt = "let" ws name ws "=" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_let_stmt(self, node, ch):
        var_name = ch[2]
        expr_node = _expr(ch[6])
        return Let(var_name, expr_node)

    # ----------------------------------------------------------------
    # Expression Statement: expr_stmt = expr ";" ws
    # ----------------------------------------------------------------
    def visit_expr_stmt(self, node, ch):
        return _expr(ch[0])

    # ----------------------------------------------------------------
    # Function Definition (MODIFIED TO USE TUPLES FOR PARAMS)
    # func_def = "def" ws name ws "(" ws params? ws ")" ws ("->" ws type)? ws "{" ws func_body "}" ws
    # ----------------------------------------------------------------
    def visit_func_def(self, node, visited_children):
        func_name = visited_children[2]
        param_info_list = []
        params_node_result = visited_children[6]
        if isinstance(params_node_result, list) and len(params_node_result) > 0:
            potential_params = params_node_result[0]
            if isinstance(potential_params, list) and all(isinstance(p, tuple) for p in potential_params):
                 param_info_list = potential_params

        # Convert lists to tuples to match expected AST structure
        param_names = tuple([name for name, type in param_info_list]) # Now a tuple
        param_types = tuple([type for name, type in param_info_list]) # Now a tuple

        return_type = MatrixType((ConcreteDim(0), ConcreteDim(0)))
        return_type_group_result = visited_children[10]

        if isinstance(return_type_group_result, list) and len(return_type_group_result) == 1 and isinstance(return_type_group_result[0], list):
            inner_list = return_type_group_result[0]
            found_type_node = None
            for item in inner_list:
                 if isinstance(item, Type):
                      found_type_node = item
                      break
            if found_type_node:
                return_type = found_type_node
            else:
                raise Exception(f"visit_func_def: Optional return type group's inner list did not contain a Type object: {inner_list}")

        func_body_node = visited_children[14]
        if isinstance(func_body_node, Block):
            func_body = func_body_node
        else:
             func_body = Block(stmts=[])

        # Pass tuples to constructors
        func_type = FunctionType(params=param_types, ret=return_type)
        return FunctionDec(name=func_name, params=param_names, body=func_body, ty=func_type)

    # ----------------------------------------------------------------
    # Parameters List: params = param ("," ws param)*
    # ----------------------------------------------------------------
    def visit_params(self, node, visited_children):
        first_param_result = visited_children[0]
        if not (isinstance(first_param_result, tuple) and len(first_param_result) == 2):
             raise Exception(f"visit_params: Expected tuple for first_param_result, got {type(first_param_result)}: {first_param_result}")
        rest_params = []
        for group in visited_children[1]:
            param_result = group[2]
            if not (isinstance(param_result, tuple) and len(param_result) == 2):
                 raise Exception(f"visit_params: Expected tuple for subsequent param_result, got {type(param_result)}: {param_result}")
            rest_params.append(param_result)
        all_params = [first_param_result] + rest_params
        return all_params # Returns list of tuples

    # ----------------------------------------------------------------
    # Parameter: param = name ws ":" ws type
    # ----------------------------------------------------------------
    def visit_param(self, node, visited_children):
        param_name = visited_children[0]
        param_type = visited_children[4]
        if not isinstance(param_name, str):
            raise Exception(f"visit_param: Expected string for parameter name, got {type(param_name)}: {param_name}")
        if not isinstance(param_type, Type):
            type_node_text = node.children[4].text
            raise Exception(f"visit_param: Expected Type for parameter type (for '{param_name}'), got {type(param_type)}: {param_type}. Parsed type text was: '{type_node_text}'")
        result = (param_name, param_type)
        return result

    # ----------------------------------------------------------------
    # Type: type = "Mat(" ws dim ws "," ws dim ws ")" ws
    # ----------------------------------------------------------------
    def visit_type(self, node, visited_children):
        def extract_dim(val):
            val = _first_expr(val) # Unwrap potential lists
            if isinstance(val, int):
                return ConcreteDim(val)
            if isinstance(val, str):
                if val.isalpha():
                     if 'TypeVarDim' not in globals() and 'TypeVarDim' not in locals():
                          # Attempt to import dynamically if needed (might indicate setup issue)
                          try:
                              from typ import TypeVarDim
                          except ImportError:
                              raise NameError("TypeVarDim class is not defined or imported")
                     return TypeVarDim(val)
                elif val.isdigit():
                     return ConcreteDim(int(val))
                else:
                     raise ValueError(f"Invalid dimension value: {val}")
            raise TypeError(f"Unexpected type for dimension value: {type(val)}")

        rows = extract_dim(visited_children[2])
        cols = extract_dim(visited_children[6])
        return MatrixType((rows, cols))

    # ----------------------------------------------------------------
    # Dim: dim = name / number
    # ----------------------------------------------------------------
    def visit_dim(self, node, visited_children):
        return visited_children[0]

    # ----------------------------------------------------------------
    # Comments / Empty Lines
    # ----------------------------------------------------------------
    def visit_comment(self, node, visited_children):
        return None
    def visit_emptyline(self, node, visited_children):
        return None

    # ----------------------------------------------------------------
    # Function Body: func_body = (statement / comment / emptyline)*
    # ----------------------------------------------------------------
    def visit_func_body(self, node, visited_children):
        statements = []
        for item in visited_children:
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
        expr = _expr(visited_children[2])
        return Return(expr)

    # ----------------------------------------------------------------
    # Addition/Subtraction: add = mul (("+" / "-") ws mul)*
    # ----------------------------------------------------------------
    def visit_add(self, node, visited_children):
        left = _expr(visited_children[0])
        for group in visited_children[1]:
            op_result = group[0]
            right = _expr(group[2])
            op_token = None
            if isinstance(op_result, str):
                 op_token = op_result.strip()
            elif isinstance(op_result, list) and len(op_result) == 1 and isinstance(op_result[0], str):
                 op_token = op_result[0].strip()
            else:
                 raise Exception(f"Could not determine operator token from visit result: {op_result}")

            if op_token == "+": op_enum = BinOp.PLUS
            elif op_token == "-": op_enum = BinOp.MINUS
            else: raise Exception(f"Unknown operator token in add: '{op_token}'")
            left = _expr(left) # Ensure left is Expr node
            left = BinaryExpr(left, right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Multiplication: mul = atom (("*") ws atom)*
    # ----------------------------------------------------------------
    def visit_mul(self, node, visited_children):
        left = _expr(visited_children[0])
        for group in visited_children[1]:
            op_result = group[0]
            right = _expr(group[2])
            op_token = None
            if isinstance(op_result, str):
                 op_token = op_result.strip()
            elif isinstance(op_result, list) and len(op_result) == 1 and isinstance(op_result[0], str):
                 op_token = op_result[0].strip()
            else:
                 raise Exception(f"Could not determine operator token from visit result: {op_result}")

            if op_token == "*": op_enum = BinOp.TIMES
            else: raise Exception(f"Unknown operator token in mul: '{op_token}'")
            left = _expr(left) # Ensure left is Expr node
            left = BinaryExpr(left, right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Atom: atom = matrix / vector / number_literal / func_call / var
    # ----------------------------------------------------------------
    def visit_atom(self, node, visited_children):
        return _expr(visited_children[0])

    # ----------------------------------------------------------------
    # Variable: var = name
    # ----------------------------------------------------------------
    def visit_var(self, node, visited_children):
        return Variable(visited_children[0])

    # ----------------------------------------------------------------
    # Number Literal: number_literal = number
    # ----------------------------------------------------------------
    def visit_number_literal(self, node, visited_children):
        num_val = visited_children[0]
        if isinstance(num_val, int):
            return Literal(num_val)
        try:
            return Literal(int(node.text.strip()))
        except ValueError:
             raise Exception(f"Could not parse number literal: {node.text}")

    def visit_number(self, node, visited_children):
        try:
             return int(node.text.strip())
        except ValueError:
             raise Exception(f"Could not parse number: {node.text}")

    # ----------------------------------------------------------------
    # Function Call: func_call = name ws "(" ws arguments? ws ")" ws
    # ----------------------------------------------------------------
    def visit_func_call(self, node, visited_children):
        func_name = visited_children[0]
        args = []
        args_node_result = visited_children[4]
        if isinstance(args_node_result, list) and len(args_node_result) > 0:
             potential_args = args_node_result[0]
             if isinstance(potential_args, list):
                  args = [_expr(arg) for arg in potential_args]
        return FunctionCall(func_name, args)

    # ----------------------------------------------------------------
    # Arguments: arguments = expr ("," ws expr)*
    # ----------------------------------------------------------------
    def visit_arguments(self, node, visited_children):
        first = _expr(visited_children[0])
        rest = [_expr(g[2]) for g in visited_children[1]]
        return [first] + rest

    # ----------------------------------------------------------------
    # Matrix Literal: matrix = "[" ws vector ("," ws vector)* "]" ws
    # ----------------------------------------------------------------
    def visit_matrix(self, node, visited_children):
        first_vec_result = visited_children[2]
        rest_vec_results = [g[2] for g in visited_children[3]]
        vector_results = [first_vec_result] + rest_vec_results
        row_lists = []
        for v in vector_results:
            if isinstance(v, MatrixLiteral) and len(v.values) == 1:
                row_lists.append(v.values[0])
            else:
                raise Exception(f"Invalid matrix row structure: Expected single-row MatrixLiteral, got {type(v)}: {repr(v)}")
        return MatrixLiteral(values=row_lists, node_type='MatrixLiteral')


    # ----------------------------------------------------------------
    # Vector Literal: vector = "[" ws expr ("," ws expr)* "]" ws
    # ----------------------------------------------------------------
    def visit_vector(self, node, visited_children):
        first_expr = _expr(visited_children[2])
        rest_exprs = [_expr(g[2]) for g in visited_children[3]]
        elements = [first_expr] + rest_exprs
        return MatrixLiteral(values=[elements], node_type='MatrixLiteral')

    # ----------------------------------------------------------------
    # Print Statement: print_stmt = "print" ws "(" ws expr ("," ws expr)* ws ")" ws ";" ws
    # ----------------------------------------------------------------
    def visit_print_stmt(self, node, ch):
        first_arg = _expr(ch[4])
        rest_args = [_expr(g[2]) for g in ch[5]]
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
         print(f"\n!!! VISITOR ERROR processing {file_name} !!!")
         print(f"Error Type: {type(visit_error).__name__}")
         print(f"Error Args: {visit_error.args}")
         raise visit_error

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
        grammar = Grammar(Path("grammar.peg").read_text())
        tree = grammar.parse(Path(input_file).read_text())
        print("\n--- Parse Tree ---")
        print(tree)
        print("\n--- Visiting Tree to build AST ---")
        visitor = MatrixVisitor()
        ast = visitor.visit(tree)
        print("\n--- Generated AST ---")
        print(repr(ast))
        print("\nParsing and AST construction successful.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"\nAn error occurred during parsing or visiting:")
        import traceback
        traceback.print_exc()

