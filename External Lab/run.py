# run.py

import numpy as np
import sys
import os
from pathlib import Path # Use pathlib for consistency

# Assuming these modules exist and are correct now
from parse import parse
from utils import ScopedDict
from AST import interpret_block
from typ import type_block, TypeError as MatLangTypeError # Avoid conflict with built-in TypeError

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_file.mat>")
        exit(1)

    file_path_str = sys.argv[1]
    file_path = Path(file_path_str)

    if not file_path.is_file():
        print(f"Error: File not found -> {file_path_str}")
        exit(1)

    # --- Parsing ---
    print(f"Parsing file: {file_path_str}")
    ast = None
    try:
        ast = parse(file_path_str)
        print("\nAST:")
        # Use repr for potentially more detail than print(ast)
        print(repr(ast))
    except FileNotFoundError as e:
        print(f"Error during parsing setup: {e}")
        exit(1)
    except Exception as e:
        # Parser should print specific errors, but catch any others
        print(f"Unexpected error during parsing stage: {e}")
        # Optionally print full traceback for debugging
        # import traceback
        # traceback.print_exc()
        exit(1) # Stop if parsing fails

    # --- Type Checking ---
    print("\nType Checking...")
    type_bindings = ScopedDict()
    type_declarations = ScopedDict()
    try:
        type_block(ast, type_bindings, type_declarations)
        print("Type checking successful.")
    except MatLangTypeError as e:
        print(f"\n!!! TYPE ERROR !!!")
        print(f"Error: {e.msg}")
        # Potentially exit if type errors should halt execution
        # exit(1) # For assignment, maybe continue to interpreter if needed?
        # Let's exit on type error as per typical compiler behavior
        exit(1)
    except Exception as e:
        print(f"\n!!! Unexpected error during type checking !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Args: {e.args}")
        # import traceback
        # traceback.print_exc()
        exit(1)

    # --- Interpretation ---
    print("\nInterpreting...")
    interp_bindings = ScopedDict()
    # Populate interp_declarations from type_declarations (which holds FunctionDec nodes)
    interp_declarations = ScopedDict()
    # Need to access the underlying dicts if ScopedDict doesn't support items() directly
    for scope in type_declarations.dicts:
         for name, decl in scope.items():
              interp_declarations[name] = decl # Assuming FunctionDec nodes are stored

    final_value = None
    try:
        # Interpret the AST
        final_value = interpret_block(ast, interp_bindings, interp_declarations)
        print("\nInterpretation finished.")
    except MatLangTypeError as e: # Catch TypeErrors that might occur at runtime if checker missed something
        print(f"\n!!! RUNTIME TYPE ERROR !!!")
        print(f"Error: {e.msg}")
        exit(1)
    except KeyError as e: # Catch undefined variable errors from ScopedDict
        print(f"\n!!! RUNTIME ERROR: Undefined variable {e} !!!")
        exit(1)
    except Exception as e:
        print(f"\n!!! Unexpected error during interpretation !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Args: {e.args}")
        # import traceback
        # traceback.print_exc()
        exit(1)

    # --- Print Final Result (Modified) ---
    print("\nFinal Evaluation Result (value of last statement/return):")
    # Print the actual numpy array or value returned by interpret_block
    if final_value is not None:
         # Check if it's the default empty array and print something clearer
         if isinstance(final_value, np.ndarray) and final_value.size == 0:
              print("[] (Mat(0,0))")
         else:
              print(final_value)
    else:
         # Should not happen if interpret_block always returns something
         print("<No final value returned>")


if __name__ == "__main__":
    main()

