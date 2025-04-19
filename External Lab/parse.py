# parse.py

from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import Node
from parsimonious.grammar import Grammar
from pathlib import Path
import sys
import re

# --- Explicit Imports ---
from AST import (
    Statement, Expr, Block, BinaryExpr, Let, Literal, Variable,
    MatrixLiteral, FunctionCall, FunctionDec, Return, BinOp
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
        if item in ['+', '-', '*']:
             return item
        return Variable(item)
    if isinstance(item, list) and item:
        if len(item) == 1:
             return _expr(item[0])
    if isinstance(item, int):
        return Literal(item)
    return item

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
    # Function Definition
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

        param_names = tuple([name for name, type in param_info_list])
        param_types = tuple([type for name, type in param_info_list])

        return_type = MatrixType((ConcreteDim(0), ConcreteDim(0))) # Default type uses ConcreteDim
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
        return all_params

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
    # (MODIFIED TO STORE STRINGS FOR TYPE VARS IN SHAPE)
    # ----------------------------------------------------------------
    def visit_type(self, node, visited_children):
        # This helper now returns ConcreteDim for numbers, and STRINGS for type vars
        def extract_dim_for_pickle(val):
            val = _first_expr(val) # Unwrap potential lists
            if isinstance(val, int):
                # Return ConcreteDim for numbers
                return ConcreteDim(val)
            if isinstance(val, str):
                if val.isalpha():
                     # Return the string name directly for type variables
                     return val
                elif val.isdigit():
                     # Handle numbers parsed as names (should ideally be parsed as number)
                     return ConcreteDim(int(val))
                else:
                     raise ValueError(f"Invalid dimension value: {val}")
            raise TypeError(f"Unexpected type for dimension value: {type(val)}")

        rows = extract_dim_for_pickle(visited_children[2]) # Result of visit_dim
        cols = extract_dim_for_pickle(visited_children[6]) # Result of visit_dim

        # Create MatrixType with shape containing ConcreteDim or str
        return MatrixType(shape=(rows, cols))

    # ----------------------------------------------------------------
    # Dim: dim = name / number
    # ----------------------------------------------------------------
    def visit_dim(self, node, visited_children):
        # Returns int from visit_number or str from visit_name
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
            left = _expr(left)
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
            left = _expr(left)
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
        # Ensure node_type matches if needed by __eq__ in AST.py
        # Check your MatrixLiteral definition and its __eq__ method.
        # If node_type is part of __eq__, include it here.
        # Assuming it might be needed based on previous AST printouts:
        return MatrixLiteral(values=row_lists, node_type='MatrixLiteral')


    # ----------------------------------------------------------------
    # Vector Literal: vector = "[" ws expr ("," ws expr)* "]" ws
    # ----------------------------------------------------------------
    def visit_vector(self, node, visited_children):
        first_expr = _expr(visited_children[2])
        rest_exprs = [_expr(g[2]) for g in visited_children[3]]
        elements = [first_expr] + rest_exprs
        # Ensure node_type matches if needed by __eq__ in AST.py
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
        # grammar = Grammar(Path("grammar.peg").read_text()) # Already read in parse()
        # tree = grammar.parse(Path(input_file).read_text()) # Already done in parse()
        # print("\n--- Parse Tree ---") # Printing tree can be very verbose
        # print(tree)
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

