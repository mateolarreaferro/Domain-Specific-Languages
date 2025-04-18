import numpy as np
import sys
import os
from parse import parse
from utils import ScopedDict
from AST import interpret_block
from typ import type_block

def main():
    if len(sys.argv) != 2:
        print("Usage: run.py [file]")
        exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: File not found -> {file_path}")
        exit(1)

    print(f"Parsing file: {file_path}")
    ast = parse(file_path)
    print("\nAST:")
    print(ast)

    bindings = ScopedDict()
    declarations = ScopedDict()
    type_block(ast, bindings, declarations)

    # Reinitialize for interpretation
    bindings = ScopedDict()
    declarations = ScopedDict()
    result = interpret_block(ast, bindings, declarations)

    print("\nEvaluation result:")
    print(result)

if __name__ == "__main__":
    main()
