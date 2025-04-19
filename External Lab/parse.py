from parsimonious.nodes import NodeVisitor  # Allows walking the parse tree.
from parsimonious.nodes import Node         # Represents nodes in the parse tree.
from parsimonious.grammar import Grammar    # Loads and works with the PEG grammar.
from pathlib import Path                    # Helps read files (grammar and input).
from AST import *                           # Import AST classes (Block, Let, etc.)
from typ import *                           # Import type-checking utilities
import sys                                  # Provides command-line argument support

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
    def visit_func_def(self, node, ch):
        name = ch[2]  # the identifier after 'def'

        # Harvest every (param_name, param_type) pair
        param_pairs = []

        def walk(x):
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], Type):
                param_pairs.append(x)
            elif isinstance(x, list):
                for y in x:
                    walk(y)

        walk(ch)

        param_names = [n for n, _ in param_pairs]
        param_types = [t for _, t in param_pairs]

        # Find return type (first MatrixType child, else implicit Mat(0,0))
        ret_ty = next((c for c in ch if isinstance(c, MatrixType)),
                      MatrixType((ConcreteDim(0), ConcreteDim(0))))

        # Find body (first Block child)
        body = next(c for c in ch if isinstance(c, Block))

        return FunctionDec(
            name,
            param_names,
            body,
            FunctionType(param_types, ret_ty),
        )

    # ----------------------------------------------------------------
    # Parameters List:
    # params = param ("," ws param)*  
    # ----------------------------------------------------------------
    def visit_params(self, node, visited_children):
        param_list = [visited_children[0]]
        for group in visited_children[1]:
            param_list.append(group[2])
        return param_list

    # ----------------------------------------------------------------
    # Parameter:
    # param = name ws ":" ws type
    # ----------------------------------------------------------------
    def visit_param(self, node, visited_children):
        param_name = visited_children[0]
        if isinstance(param_name, list):  # unpack if needed
            param_name = param_name[0]
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
        stmts = []
        for child in visited_children:
            if isinstance(child, list):
                for sub in child:
                    if isinstance(sub, Statement) or hasattr(sub, 'node_type'):
                        stmts.append(sub)
            elif isinstance(child, Statement) or hasattr(child, 'node_type'):
                stmts.append(child)
        return Block(stmts)

    # ----------------------------------------------------------------
    # Return Statement:
    # return_stmt = "return" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_return_stmt(self, node, visited_children):
        expr_node = _expr(visited_children[2])
        return Return(expr_node)

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
        func_name = visited_children[0]
        # Check if arguments are present
        if isinstance(visited_children[4], list) and visited_children[4]:
            args = visited_children[4]
        else:
            args = []
        return FunctionCall(func_name, args)

    # ----------------------------------------------------------------
    # Arguments:
    # arguments = expr ("," ws expr)*
    # ----------------------------------------------------------------
    def visit_arguments(self, node, visited_children):
        first = visited_children[0]                 # the first expression
        rest  = [g[2] for g in visited_children[1]] # remaining expressions

        raw_args = [first] + rest                   # list may contain Expr or [Expr]
        args = [_first_expr(a) for a in raw_args]   # flatten each entry to a real Expr
        return args

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
    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(file_name).read_text())
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