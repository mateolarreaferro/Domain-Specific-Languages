from parsimonious.nodes import NodeVisitor  # Allows walking the parse tree.
from parsimonious.nodes import Node         # Represents nodes in the parse tree.
from parsimonious.grammar import Grammar       # Loads and works with the PEG grammar.
from pathlib import Path                       # Helps read files (grammar and input).
from AST import *                              # Import AST classes (Block, LetStmt, etc.)
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
        # 'program' -> (statement / comment / emptyline)*
        # Flatten the list and filter out comments and empty lines (which are not needed for the AST)
        statements = []
        for child in visited_children:
            # Each child may itself be a list from the generic visit
            if isinstance(child, list):
                for sub in child:
                    # Only add sub-nodes that are meaningful AST nodes (e.g., not just whitespace/comments)
                    if hasattr(sub, 'node_type'):
                        statements.append(sub)
            else:
                if hasattr(child, 'node_type'):
                    statements.append(child)
        # The root AST node is a Block containing all statements.
        return Block(statements)

    # ----------------------------------------------------------------
    # Let Statement: let_stmt = "let" ws name ws "=" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_let_stmt(self, node, visited_children):
        # visited_children is expected to be:
        # ["let", ws, name, ws, "=", ws, expr, ";" , ws]
        var_name = visited_children[2]     # Extract the variable name from the third element.
        expr_node = visited_children[6]      # Extract the expression (the value) from the seventh element.
        # Create a LetStmt AST node (assumes LetStmt takes variable name and expression)
        return LetStmt(var_name, expr_node)

    # ----------------------------------------------------------------
    # Expression Statement: expr_stmt = expr ";" ws
    # ----------------------------------------------------------------
    def visit_expr_stmt(self, node, visited_children):
        # visited_children: [expr, ";" , ws]
        expr_node = visited_children[0]      # The first element is the expression.
        # Create an ExpressionStmt AST node wrapping the expression.
        return ExpressionStmt(expr_node)

    # ----------------------------------------------------------------
    # Function Definition:
    # func_def = "def" ws name ws "(" ws params? ")" ws "->" ws type ws "{" ws func_body "}" ws
    # ----------------------------------------------------------------
    def visit_func_def(self, node, visited_children):
        # Expected visited_children structure:
        # ["def", ws, name, ws, "(", ws, params? (or []), ")" , ws, "->", ws, type, ws, "{", ws, func_body, "}", ws]
        func_name = visited_children[2]      # The function name.
        params = visited_children[6] if visited_children[6] != [] else []  # Parameters (may be empty)
        return_type = visited_children[10]     # The type annotation for the return value.
        func_body = visited_children[14]       # The function body.
        # Create a FunctionDef AST node with the function name, parameters, return type, and body.
        return FunctionDef(func_name, params, return_type, func_body)

    # ----------------------------------------------------------------
    # Parameters List:
    # params = param ("," ws param)*  
    # ----------------------------------------------------------------
    def visit_params(self, node, visited_children):
        # visited_children: [param, ("," ws param)*]
        param_list = [visited_children[0]]  # Start with the first parameter.
        # For each additional parameter group, extract the parameter (third element of each group)
        for group in visited_children[1]:
            # Each group is expected to be a list like [",", ws, param]
            param_list.append(group[2])
        return param_list

    # ----------------------------------------------------------------
    # Parameter:
    # param = name ws ":" ws type
    # ----------------------------------------------------------------
    def visit_param(self, node, visited_children):
        # visited_children: [name, ws, ":", ws, type]
        param_name = visited_children[0]   # Extract the parameter name.
        param_type = visited_children[4]   # Extract the type.
        # Create a Parameter AST node.
        return Parameter(param_name, param_type)

    # ----------------------------------------------------------------
    # Type:
    # type = "Mat(" ws number ws "," ws number ws ")" ws
    # ----------------------------------------------------------------
    def visit_type(self, node, visited_children):
        # visited_children: ["Mat(", ws, number, ws, ",", ws, number, ws, ")"]
        rows_token = visited_children[2]   # The number of rows as text.
        cols_token = visited_children[6]   # The number of columns as text.
        # Convert the string tokens to integers.
        rows = int(rows_token)
        cols = int(cols_token)
        # Create a MatrixType AST node (or similar) representing Mat(rows, cols)
        return MatrixType(rows, cols)

    # ----------------------------------------------------------------
    # Function Body:
    # func_body = (statement)* return_stmt?
    # ----------------------------------------------------------------
    def visit_func_body(self, node, visited_children):
        # visited_children: a list where each element is a statement, and optionally a return statement at the end
        stmts = []
        for child in visited_children:
            if isinstance(child, list):
                for sub in child:
                    if hasattr(sub, 'node_type'):
                        stmts.append(sub)
            else:
                if hasattr(child, 'node_type'):
                    stmts.append(child)
        # Return a Block AST node for the function body.
        return Block(stmts)

    # ----------------------------------------------------------------
    # Return Statement:
    # return_stmt = "return" ws expr ";" ws
    # ----------------------------------------------------------------
    def visit_return_stmt(self, node, visited_children):
        # visited_children: ["return", ws, expr, ";" , ws]
        expr_node = visited_children[2]      # Extract the expression after 'return'
        return ReturnStmt(expr_node)

    # ----------------------------------------------------------------
    # Expression
    # expr = add
    # ----------------------------------------------------------------
    def visit_expr(self, node, visited_children):
        # 'expr' is simply the same as 'add'; return the result from add.
        return visited_children[0]

    # ----------------------------------------------------------------
    # Addition/Subtraction:
    # add = mul (("+" / "-") ws mul)*
    # ----------------------------------------------------------------
    def visit_add(self, node, visited_children):
        # visited_children: first element is a 'mul' node, then a list of groups where each group is [operator, ws, mul]
        left = visited_children[0]
        for group in visited_children[1]:
            op = group[0]                # The operator: '+' or '-'
            right = group[2]             # The next multiplication expression
            # Combine into a BinaryOp AST node that represents left op right
            left = BinaryOp(op, left, right)
        return left

    # ----------------------------------------------------------------
    # Multiplication:
    # mul = atom (("*") ws atom)*
    # ----------------------------------------------------------------
    def visit_mul(self, node, visited_children):
        # visited_children: first element is an 'atom', then zero or more groups [ "*", ws, atom ]
        left = visited_children[0]
        for group in visited_children[1]:
            op = group[0]                # The '*' operator
            right = group[2]             # The subsequent 'atom'
            left = BinaryOp(op, left, right)
        return left

    # ----------------------------------------------------------------
    # Atom:
    # atom = matrix / vector / number_literal / func_call / var
    # ----------------------------------------------------------------
    def visit_atom(self, node, visited_children):
        # visited_children: should contain a single element corresponding to one of the alternatives.
        return visited_children[0]

    # ----------------------------------------------------------------
    # Variable:
    # var = name
    # ----------------------------------------------------------------
    def visit_var(self, node, visited_children):
        # 'var' directly uses the 'name' rule.
        var_name = visited_children[0]
        return Var(var_name)

    # ----------------------------------------------------------------
    # Number Literal:
    # number_literal = number
    # ----------------------------------------------------------------
    def visit_number_literal(self, node, visited_children):
        # visited_children contains the token from the 'number' rule.
        number_text = visited_children[0]
        return NumberLiteral(int(number_text))

    # ----------------------------------------------------------------
    # Function Call:
    # func_call = name ws "(" ws arguments? ")" ws
    # ----------------------------------------------------------------
    def visit_func_call(self, node, visited_children):
        # visited_children: [name, ws, "(", ws, arguments?, ")" , ws]
        func_name = visited_children[0]
        args = visited_children[4] if visited_children[4] != [] else []
        return FuncCall(func_name, args)

    # ----------------------------------------------------------------
    # Arguments:
    # arguments = expr ("," ws expr)*
    # ----------------------------------------------------------------
    def visit_arguments(self, node, visited_children):
        # visited_children: first element is an expression, then a list of groups [",", ws, expr]
        arg_list = [visited_children[0]]
        for group in visited_children[1]:
            arg_list.append(group[2])
        return arg_list

    # ----------------------------------------------------------------
    # Matrix Literal:
    # matrix = "[" ws vector ("," ws vector)* "]" ws
    # ----------------------------------------------------------------
    def visit_matrix(self, node, visited_children):
        # visited_children: ["[", ws, vector, ("," ws vector)*, "]", ws]
        first_vector = visited_children[2]
        vectors = [first_vector]
        for group in visited_children[3]:
            # Each group is a list: [",", ws, vector]
            vectors.append(group[2])
        return MatrixLiteral(vectors)

    # ----------------------------------------------------------------
    # Vector Literal:
    # vector = "[" ws expr ("," ws expr)* "]" ws
    # ----------------------------------------------------------------
    def visit_vector(self, node, visited_children):
        # visited_children: ["[", ws, expr, ("," ws expr)*, "]", ws]
        first_expr = visited_children[2]
        elements = [first_expr]
        for group in visited_children[3]:
            # Each group: [",", ws, expr]
            elements.append(group[2])
        return VectorLiteral(elements)

# ----------------------------------------------------------------
# parse() function: Reads a file, parses it using the grammar, and returns an AST.
# ----------------------------------------------------------------
def parse(file_name: str):
    # Load the grammar from 'grammar.peg'
    grammar = Grammar(Path("grammar.peg").read_text())
    # Read the MatLang source file and parse it to generate a parse tree.
    tree = grammar.parse(Path(file_name).read_text())
    # Create a MatrixVisitor to convert the parse tree to an AST.
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
    # Load the grammar and parse the provided input file.
    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(sys.argv[1]).read_text())
    # For now, print the raw parse tree (or later, you can print the AST)
    print(tree)
