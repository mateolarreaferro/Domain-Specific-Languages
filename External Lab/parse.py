from parsimonious.nodes import NodeVisitor  # Allows walking the parse tree.
from parsimonious.nodes import Node         # Represents nodes in the parse tree.
from parsimonious.grammar import Grammar    # Loads and works with the PEG grammar.
from pathlib import Path                    # Helps read files (grammar and input).
from AST import *                           # Import AST classes (Block, Let, etc.)
from typ import *                           # Import type-checking utilities
import sys                                  # Provides command-line argument support
import re

# ────────────────────────────────────────────────────────────────────────────
# Helper functions for expression handling
# ────────────────────────────────────────────────────────────────────────────
def _expr(item):
    """Return an Expr; wrap bare identifier strings as Variable."""
    if isinstance(item, Expr):
        return item
    if isinstance(item, str):
        return Variable(item)
    if isinstance(item, list) and item:
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

class MatrixVisitor(NodeVisitor):
    """Visitor for the parse tree that converts a MatLang parse tree (as defined in grammar.peg)
    into an Abstract Syntax Tree (AST). Each visit method corresponds to a grammar rule.
    """

    def generic_visit(self, node, visited_children):
        # Default method that returns the visited children if present,
        # or the node text otherwise.
        return visited_children or node.text
    
    def visit_name(self, node, visited_children):
        # strip trailing ws and return plain string
        return node.text.strip()

    # ----------------------------------------------------------------
    # Program
    # ----------------------------------------------------------------
    def visit_program(self, node, visited_children):
        def flatten_statements(children):
            result = []
            for child in children:
                if isinstance(child, list):
                    result.extend(flatten_statements(child))
                elif isinstance(child, Statement):
                    result.append(child)
            return result

        statements = flatten_statements(visited_children)
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
        return ExpressionStatement(_expr(ch[0]))

    # ----------------------------------------------------------------
    # Function Definition
    # ----------------------------------------------------------------
    def visit_func_def(self, node, visited_children):
        """Process function definition"""
        # print("\n--- visit_func_def ---") # Optional Debugging
        # for i, child in enumerate(visited_children):
        #     print(f"Index {i}: Type={type(child)}, Value={repr(child)}")
        # print("----------------------\n")

        # Indices (approximate):
        # "def" ws name ws "(" ws params? ws ")" ws ("->" ws type)? ws "{" ws func_body "}" ws
        #  0    1   2   3   4   5    6     7   8   9       10        11 12 13     14    15 16
        func_name = visited_children[2]

        param_info_list = [] # Default: no parameters
        params_node_result = visited_children[6]
        if isinstance(params_node_result, list) and len(params_node_result) > 0:
            potential_params = params_node_result[0]
            if isinstance(potential_params, list) and all(isinstance(p, tuple) for p in potential_params):
                param_info_list = potential_params

        param_names = [name for name, type in param_info_list]
        param_types = [type for name, type in param_info_list]

        # --- Updated Return Type Handling Again ---
        return_type = MatrixType((ConcreteDim(0), ConcreteDim(0))) # Default: Mat(0,0)
        return_type_group_result = visited_children[10] # This corresponds to the ("->" ws type)? group

        # Check if the optional group matched. It often returns [['->', ws, type_node]]
        if isinstance(return_type_group_result, list) and len(return_type_group_result) == 1 and isinstance(return_type_group_result[0], list):
            # Get the inner list containing the visited children of "->" ws type
            inner_list = return_type_group_result[0]
            # print(f"Inner list for return type: {inner_list}") # Optional Debugging

            found_type_node = None
            # Iterate through the INNER list to find the Type object
            for item in inner_list:
                if isinstance(item, Type):
                    found_type_node = item
                    break # Found the type node

            if found_type_node:
                return_type = found_type_node
            else:
                raise Exception(f"visit_func_def: Optional return type group's inner list did not contain a Type object: {inner_list}")

        # --- End Updated Return Type Handling ---

        func_body_node = visited_children[14]
        if isinstance(func_body_node, Block):
            func_body = func_body_node
        else:
            # print(f"Warning: func_body was not a Block node (index 14). Type: {type(func_body_node)}. Assuming empty body.")
            func_body = Block(stmts=[])

        func_type = FunctionType(params=param_types, ret=return_type)

        return FunctionDec(name=func_name, params=param_names, body=func_body, ty=func_type)
    
    # ----------------------------------------------------------------
# Parameter: param = name ws ":" ws type
# ----------------------------------------------------------------
    def visit_param(self, node, visited_children):
        """Process single parameter with type. Returns (name: str, type: Type)"""
        # print(f"visit_param children: {visited_children}") # Optional Debugging
        # Indices based on rule: name[0] ws[1] :[2] ws[3] type[4]
        param_name = visited_children[0]
        param_type = visited_children[4]

        # Ensure param_name is a string (result of visit_name)
        if not isinstance(param_name, str):
            raise Exception(f"visit_param: Expected string for parameter name, got {type(param_name)}: {param_name}")

        # Ensure param_type is a Type object (result of visit_type)
        if not isinstance(param_type, Type):
            # It's possible visit_type failed or returned something unexpected.
            # Let's check the type node directly in the parse tree for clues.
            type_node_text = node.children[4].text # Get the text of the type part
            raise Exception(f"visit_param: Expected Type for parameter type (for '{param_name}'), got {type(param_type)}: {param_type}. Parsed type text was: '{type_node_text}'")

        result = (param_name, param_type)
        # print(f"visit_param returning: {result}") # Optional Debugging
        return result

# ----------------------------------------------------------------
# Parameters List: params = param ("," ws param)*
# ----------------------------------------------------------------
    def visit_params(self, node, visited_children):
        """Process function parameter list. Returns a list of (name, type) tuples."""
        # print(f"visit_params children: {visited_children}") # Optional Debugging
        # visited_children structure: [param_result, list_of_groups]
        # where param_result is from visit_param, and each group is [",", ws, param_result]

        first_param_result = visited_children[0]

        # Validate the first parameter's structure IMMEDIATELY
        if not (isinstance(first_param_result, tuple) and len(first_param_result) == 2):
            raise Exception(f"visit_params: Expected tuple for first_param_result, got {type(first_param_result)}: {first_param_result}")

        # Process the rest of the parameters
        rest_params = []
        for group in visited_children[1]:
            param_result = group[2] # Get the param result from the group [",", ws, param_result]
            # Validate each subsequent parameter's structure
            if not (isinstance(param_result, tuple) and len(param_result) == 2):
                raise Exception(f"visit_params: Expected tuple for subsequent param_result, got {type(param_result)}: {param_result}")
            rest_params.append(param_result)

        # Combine the first parameter and the rest
        all_params = [first_param_result] + rest_params
        # print(f"visit_params returning: {all_params}") # Optional Debugging
        return all_params # List of (name, type) tuples

    # ----------------------------------------------------------------
    # Type:
    # type = "Mat(" ws dim ws "," ws dim ws ")" ws
    # ----------------------------------------------------------------
    def visit_type(self, node, visited_children):
        def extract_dim(val):
            if isinstance(val, list):
                return extract_dim(val[0])
            if isinstance(val, int):
                return ConcreteDim(val)
            if isinstance(val, str):
                return TypeVarDim(val) if val.isalpha() else ConcreteDim(int(val))
            return TypeVarDim(str(val))

        rows = extract_dim(visited_children[2])
        cols = extract_dim(visited_children[6])
        return MatrixType((rows, cols))
    
    # ----------------------------------------------------------------
    # Dim:
    # dim = name / number
    # ----------------------------------------------------------------
    def visit_dim(self, node, visited_children):
        val = visited_children[0]
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return val
    
    def visit_comment(self, node, visited_children):
        return None

    def visit_emptyline(self, node, visited_children):
        return None

    # ----------------------------------------------------------------
    # Function Body:
    # func_body = (statement / comment / emptyline)* return_stmt?
    # ----------------------------------------------------------------
    def visit_func_body(self, node, visited_children):
        """Process function body with statements and optional return statement"""
        statements = []
        
        # Process all regular statements
        for child in visited_children:
            if isinstance(child, Statement):
                statements.append(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Statement):
                        statements.append(item)
                    elif isinstance(item, list):
                        for subitem in item:
                            if isinstance(subitem, Statement):
                                statements.append(subitem)
        
        return Block(statements)

    # ----------------------------------------------------------------
    # Return Statement:
    # return_stmt = "return" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_return_stmt(self, node, visited_children):
        """Process a return statement"""
        # Expression is at index 2
        expr = _expr(visited_children[2])
        return Return(expr)

    # ----------------------------------------------------------------
    # Addition/Subtraction:
    # add = mul (("+" / "-") ws mul)*
    # ----------------------------------------------------------------
    def visit_add(self, node, visited_children):
        left = visited_children[0]
        for group in visited_children[1]:
            op_token = group[0][0] if isinstance(group[0], list) else group[0]
            right = _expr(group[2])
            if op_token == "+":
                op_enum = BinOp.PLUS
            elif op_token == "-":
                op_enum = BinOp.MINUS
            else:
                raise Exception(f"Unknown operator in add: {op_token}")
            left = BinaryExpr(_expr(left), right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Multiplication:
    # mul = atom (("*") ws atom)*
    # ----------------------------------------------------------------
    def visit_mul(self, node, visited_children):
        left = visited_children[0]
        for group in visited_children[1]:
            op_token = group[0][0] if isinstance(group[0], list) else group[0]
            right = _expr(group[2])
            if op_token == "*":
                op_enum = BinOp.TIMES
            else:
                raise Exception(f"Unknown operator in mul: {op_token}")
            left = BinaryExpr(_expr(left), right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Atom:
    # atom = matrix / vector / number_literal / func_call / var
    # ----------------------------------------------------------------
    def visit_atom(self, node, visited_children):
        return _expr(visited_children[0])

    # ----------------------------------------------------------------
    # Variable:
    # var = name
    # ----------------------------------------------------------------
    def visit_var(self, node, visited_children):
        return Variable(visited_children[0])
    
    def visit_number(self, node, visited_children):
        # number is the TOKEN rule ~r"[0-9]+"
        return int(node.text.strip())

    # ----------------------------------------------------------------
    # Number Literal:
    # number_literal = number
    # ----------------------------------------------------------------
    def visit_number_literal(self, node, visited_children):
        num_val = visited_children[0]
        if isinstance(num_val, int):
            return Literal(num_val)
        return Literal(int(node.text.strip()))

    # ----------------------------------------------------------------
    # Function Call:
    # func_call = name ws "(" ws arguments? ")" ws
    # ----------------------------------------------------------------
    def visit_func_call(self, node, visited_children):
        """Process function call expressions"""
        func_name = visited_children[0]
        
        # Extract arguments
        args = []
        
        # Arguments are at index 4
        if len(visited_children) > 4 and visited_children[4]:
            arg_list = visited_children[4]
            if isinstance(arg_list, list):
                # Single argument
                if len(arg_list) == 1:
                    args.append(_expr(arg_list[0]))
                # Multiple arguments
                elif len(arg_list) > 1:
                    first_arg = _expr(arg_list[0])
                    args.append(first_arg)
                    
                    for arg_group in arg_list[1:]:
                        if isinstance(arg_group, list) and len(arg_group) > 2:
                            next_arg = _expr(arg_group[2])
                            args.append(next_arg)
        
        return FunctionCall(func_name, args)

    # ----------------------------------------------------------------
    # Arguments:
    # arguments = expr ("," ws expr)*
    # ----------------------------------------------------------------
    def visit_arguments(self, node, visited_children):
        """Process function call arguments list"""
        first = _expr(visited_children[0])          # the first expression
        rest = [_expr(g[2]) for g in visited_children[1]]  # remaining expressions
        
        # Return a flat list of expressions, not nested lists
        return [first] + rest

    # ----------------------------------------------------------------
    # Matrix Literal:
    # matrix = "[" ws vector ("," ws vector)* "]" ws
    # ----------------------------------------------------------------
    def visit_matrix(self, node, visited_children):
        first_vec = visited_children[2]
        rest_vecs = [g[2] for g in visited_children[3]]
        vectors = [first_vec] + rest_vecs

        # Expect each row to be a VectorLiteral; store its element list
        row_lists = []
        for v in vectors:
            if isinstance(v, VectorLiteral):
                row_lists.append(v.elements)
            else:
                raise Exception(f"Invalid matrix row: {v}")
        return MatrixLiteral(row_lists)

    # ----------------------------------------------------------------
    # Vector Literal:
    # vector = "[" ws expr ("," ws expr)* "]" ws
    # ----------------------------------------------------------------
    def visit_vector(self, node, visited_children):
        first_expr = _expr(visited_children[2])
        rest_exprs = [_expr(g[2]) for g in visited_children[3]]
        return VectorLiteral([first_expr] + rest_exprs)

    # ----------------------------------------------------------------
    # Print Statement:
    # print_stmt = "print" ws "(" ws expr ("," ws expr)* ws ")" ws ";" ws
    # ----------------------------------------------------------------
    def visit_print_stmt(self, node, ch):
        first = _expr(ch[4])
        rest = [_expr(g[2]) for g in ch[5]]
        return Print([first] + rest)

# ----------------------------------------------------------------
# parse() function: Reads a file, parses it using the grammar, and returns an AST.
# ----------------------------------------------------------------
def parse(file_name: str):
    """Parse a MatLang file and return the AST"""
    # Read the grammar and file
    grammar = Grammar(Path("grammar.peg").read_text())
    source_text = Path(file_name).read_text()
    
    # Parse using the grammar
    tree = grammar.parse(source_text)
    visitor = MatrixVisitor()
    ast = visitor.visit(tree)
    
    return ast

# ----------------------------------------------------------------
# Command-Line Entry Point: Allows testing by running "python parse.py [input_file]"
# ----------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse.py [file]")
        exit(1)
    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(sys.argv[1]).read_text())
    print(tree)