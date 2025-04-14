from parse import parse
import pickle


if __name__ == "__main__":
    # AST tests

    examples = ["examples/associativity.mat",
     "examples/expression.mat",
     "examples/function.mat",
     "examples/let.mat",
     "examples/precedence.mat"]
    asts = {}
    for example in examples:
        asts[example] = parse(example)

    with open("sample_asts.pickle", "rb") as file:
        correct_asts = pickle.load(file)

    for example in examples:
        assert asts[example] == correct_asts[example], f"{example} failed"

    print("All tests passed!")
