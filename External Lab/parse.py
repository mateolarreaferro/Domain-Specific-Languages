from parsimonious.nodes import NodeVisitor  # Allows walking the parse tree.
from parsimonious.nodes import Node         # Represents nodes in the parse tree.
from parsimonious.grammar import Grammar       # Loads and works with the PEG grammar.
from pathlib import Path                       # Helps read files (grammar and input).
from AST import *                              # Import AST classes (Block, Let, etc.)
from typ import *                              # Import type-checking utilities (if needed)
import sys                                     # Provides command-line argument support


class MatrixVisitor(NodeVisitor):
    """Visitor for the parse tree that converts a MatLang parse tree (as defined in grammar.peg)
    into an Abstract Syntax Tree (AST). Each visit method corresponds to a grammar rule.
    """

    def generic_visit(self, node, visited_children):
        # Default method that returns the visited children if present,
        # or the node text otherwise.
        return visited_children or node.text

    # ----------------------------------------------------------------
    # Program
    # ----------------------------------------------------------------
    def visit_program(self, node, visited_children):
        statements = []
        for child in visited_children:
            if isinstance(child, list):
                for sub in child:
                    if isinstance(sub, Statement):
                        statements.append(sub)
            elif isinstance(child, Statement):
                statements.append(child)
        return Block(statements)


    # ----------------------------------------------------------------
    # Let Statement: let_stmt = "let" ws name ws "=" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_let_stmt(self, node, visited_children):
        var_name = visited_children[2]
        expr_node = visited_children[6]
        return Let(var_name, expr_node)
    
    

    # ----------------------------------------------------------------
    # Expression Statement: expr_stmt = expr ";" ws
    # ----------------------------------------------------------------
    def visit_expr_stmt(self, node, visited_children):
        expr_node = visited_children[0]
        return ExpressionStatement(expr_node)


    # ----------------------------------------------------------------
    # Function Definition:
    # func_def = "def" ws name ws "(" ws params? ")" ws "->" ws type ws "{" ws func_body "}" ws
    # ----------------------------------------------------------------
    def visit_func_def(self, node, visited_children):
        func_name = visited_children[2]
        params = visited_children[6] if visited_children[6] != [] else []
        return_type = visited_children[10]
        func_body = visited_children[14]
        param_names = [p[0] for p in params]
        return FunctionDec(func_name, param_names, func_body, return_type)

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
        param_type = visited_children[4]
        return (param_name, param_type)

    # ----------------------------------------------------------------
    # Type:
    # type = "Mat(" ws number ws "," ws number ws ")" ws
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

    
    def visit_comment(self, node, visited_children):
        return None

    def visit_emptyline(self, node, visited_children):
        return None


    # ----------------------------------------------------------------
    # Function Body:
    # func_body = (statement)* return_stmt?
    # ----------------------------------------------------------------
    def visit_func_body(self, node, visited_children):
        stmts = []
        for child in visited_children:
            if isinstance(child, list):
                for sub in child:
                    if hasattr(sub, 'node_type') or isinstance(sub, Statement):
                        stmts.append(sub)
            else:
                if hasattr(child, 'node_type') or isinstance(child, Statement):
                    stmts.append(child)
        return Block(stmts)

    # ----------------------------------------------------------------
    # Return Statement:
    # return_stmt = "return" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_return_stmt(self, node, visited_children):
        expr_node = visited_children[2]
        return Return(expr_node)

    # ----------------------------------------------------------------
    # Expression
    # ----------------------------------------------------------------
    def visit_expr(self, node, visited_children):
        return visited_children[0]

    # ----------------------------------------------------------------
    # Addition/Subtraction:
    # add = mul (("+" / "-") ws mul)*
    # ----------------------------------------------------------------
    def visit_add(self, node, visited_children):
        left = visited_children[0]
        for group in visited_children[1]:
            op_token = group[0][0] if isinstance(group[0], list) else group[0]
            right = group[2]
            if op_token == "+":
                op_enum = BinOp.PLUS
            elif op_token == "-":
                op_enum = BinOp.MINUS
            else:
                raise Exception(f"Unknown operator in add: {op_token}")
            left = BinaryExpr(left, right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Multiplication:
    # mul = atom (("*") ws atom)*
    # ----------------------------------------------------------------
    def visit_mul(self, node, visited_children):
        left = visited_children[0]
        for group in visited_children[1]:
            op_token = group[0][0] if isinstance(group[0], list) else group[0]
            right = group[2]
            if op_token == "*":
                op_enum = BinOp.TIMES
            else:
                raise Exception(f"Unknown operator in mul: {op_token}")
            left = BinaryExpr(left, right, op_enum)
        return left

    # ----------------------------------------------------------------
    # Atom:
    # atom = matrix / vector / number_literal / func_call / var
    # ----------------------------------------------------------------
    def visit_atom(self, node, visited_children):
        return visited_children[0]

    # ----------------------------------------------------------------
    # Variable:
    # var = name
    # ----------------------------------------------------------------
    def visit_var(self, node, visited_children):
        var_name = visited_children[0]
        return Variable(var_name)

    # ----------------------------------------------------------------
    # Number Literal:
    # number_literal = number
    # ----------------------------------------------------------------
    def visit_number_literal(self, node, visited_children):
        number_text = visited_children[0]
        return Literal(int(number_text))

    # ----------------------------------------------------------------
    # Function Call:
    # func_call = name ws "(" ws arguments? ")" ws
    # ----------------------------------------------------------------
    def visit_func_call(self, node, visited_children):
        func_name = visited_children[0]
        args = visited_children[4] if visited_children[4] != [] else []
        return FunctionCall(func_name, args)

    # ----------------------------------------------------------------
    # Arguments:
    # arguments = expr ("," ws expr)*
    # ----------------------------------------------------------------
    def visit_arguments(self, node, visited_children):
        arg_list = [visited_children[0]]
        for group in visited_children[1]:
            arg_list.append(group[2])
        return arg_list

    # ----------------------------------------------------------------
    # Matrix Literal:
    # matrix = "[" ws vector ("," ws vector)* "]" ws
    # ----------------------------------------------------------------
    def visit_matrix(self, node, visited_children):
        first_vector = visited_children[2]
        vectors = [first_vector]
        for group in visited_children[3]:
            vectors.append(group[2])
        return MatrixLiteral(values=vectors)

    # ----------------------------------------------------------------
    # Vector Literal:
    # vector = "[" ws expr ("," ws expr)* "]" ws
    # ----------------------------------------------------------------
    def visit_vector(self, node, visited_children):
        first_expr = visited_children[2]
        elements = [first_expr]
        for group in visited_children[3]:
            elements.append(group[2])
        return VectorLiteral(elements)
    
    # ----------------------------------------------------------------
    # Print Statement:
    # print_stmt = "print" ws "(" ws expr ws ")" ws ";" ws
    # ----------------------------------------------------------------
    def visit_print_stmt(self, node, visited_children):
        expr_node = visited_children[4]
        return Print(expr_node)

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
