from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import Node
from parsimonious.grammar import Grammar
from pathlib import Path
from AST import *
from typ import *
import sys


class MatrixVisitor(NodeVisitor):
    """Visitor for the parse tree"""

    pass  # TODO (we have ~159 lines)

    def generic_visit(self, node, visited_children):
        """The generic visit method. Returns the node if it has no children,
        otherwise it returns the children.
        Feel free to modify this as you like.
        """
        return visited_children


def parse(file_name: str):
    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(file_name).read_text())

    visitor = MatrixVisitor()
    ast = visitor.visit(tree)
    return ast


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: parse.py [file]")
        exit(1)

    grammar = Grammar(Path("grammar.peg").read_text())
    tree = grammar.parse(Path(sys.argv[1]).read_text())
    print(tree)
