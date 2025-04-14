import numpy as np
import sys
from parse import parse
from utils import ScopedDict
from AST import interpret_block
from typ import type_block


def main():
    if len(sys.argv) != 2:
        print("Usage: run.py [file]")
        exit(1)

    ast = parse(sys.argv[1])

    bindings = ScopedDict()
    declarations = ScopedDict()
    type_block(ast, bindings, declarations)

    bindings = ScopedDict()
    declarations = ScopedDict()
    interpret_block(ast, bindings, declarations)


if __name__ == "__main__":
    main()
