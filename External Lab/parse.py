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
        # Get function name (at index 2)
        name = visited_children[2]
        
        # Extract parameters from node text directly
        func_text = node.text
        params = []
        param_types = []
        
        # Extract parameter section from the function text
        param_section_match = re.search(r'\((.*?)\)', func_text)
        if param_section_match:
            param_section = param_section_match.group(1).strip()
            if param_section:
                # Split parameters by commas
                param_items = param_section.split(',')
                for param_item in param_items:
                    param_item = param_item.strip()
                    if ':' in param_item:
                        param_parts = param_item.split(':')
                        param_name = param_parts[0].strip()
                        type_text = param_parts[1].strip()
                        
                        params.append(param_name)
                        
                        # Parse the type
                        if 'Mat(' in type_text:
                            dims_match = re.search(r'Mat\((.*?),(.*?)\)', type_text)
                            if dims_match:
                                row_dim_text = dims_match.group(1).strip()
                                col_dim_text = dims_match.group(2).strip()
                                
                                # Create dimensions
                                if row_dim_text.isalpha():
                                    row_dim = TypeVarDim(row_dim_text)
                                else:
                                    try:
                                        row_dim = ConcreteDim(int(row_dim_text))
                                    except ValueError:
                                        row_dim = TypeVarDim('a')
                                
                                if col_dim_text.isalpha():
                                    col_dim = TypeVarDim(col_dim_text)
                                else:
                                    try:
                                        col_dim = ConcreteDim(int(col_dim_text))
                                    except ValueError:
                                        col_dim = TypeVarDim('a')
                                
                                param_types.append(MatrixType((row_dim, col_dim)))
                            else:
                                # Default type if parsing fails
                                param_types.append(MatrixType((TypeVarDim('a'), TypeVarDim('a'))))
                        else:
                            # Default type if not Mat
                            param_types.append(MatrixType((TypeVarDim('a'), TypeVarDim('a'))))
        
        # Find return type (-> Type) after the params
        ret_type = MatrixType((ConcreteDim(0), ConcreteDim(0)))  # default
        ret_type_match = re.search(r'->\s*Mat\((.*?),(.*?)\)', func_text)
        if ret_type_match:
            row_dim_text = ret_type_match.group(1).strip()
            col_dim_text = ret_type_match.group(2).strip()
            
            # Create dimensions for return type
            if row_dim_text.isalpha():
                row_dim = TypeVarDim(row_dim_text)
            else:
                try:
                    row_dim = ConcreteDim(int(row_dim_text))
                except ValueError:
                    row_dim = TypeVarDim('a')
            
            if col_dim_text.isalpha():
                col_dim = TypeVarDim(col_dim_text)
            else:
                try:
                    col_dim = ConcreteDim(int(col_dim_text))
                except ValueError:
                    col_dim = TypeVarDim('a')
            
            ret_type = MatrixType((row_dim, col_dim))
        
        # Find function body (after the opening brace)
        body = Block([])  # default empty body
        for item in visited_children:
            if isinstance(item, Block):
                body = item
                break
        
        return FunctionDec(name, params, body, FunctionType(param_types, ret_type))

    # ----------------------------------------------------------------
    # Parameters List:
    # params = param ("," ws param)*  
    # ----------------------------------------------------------------
    def visit_params(self, node, visited_children):
        """Process function parameter list"""
        result = []
        
        # First parameter
        first_param = visited_children[0]
        if first_param:
            result.append(first_param)
        
        # Additional parameters (comma-separated)
        if len(visited_children) > 1 and visited_children[1]:
            for group in visited_children[1]:
                if len(group) > 2 and group[2]:
                    result.append(group[2])
        
        return result

    # ----------------------------------------------------------------
    # Parameter:
    # param = name ws ":" ws type
    # ----------------------------------------------------------------
    def visit_param(self, node, visited_children):
        """Process single parameter with type"""
        # Parameter name is at index 0
        param_name = visited_children[0]
        
        # Parameter type is at index 4
        param_type = visited_children[4]
        
        return (param_name, param_type)

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