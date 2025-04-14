from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor, Node
from enum import Enum
from pathlib import Path

grammar = Grammar(Path("grammar.peg").read_text())
parse_tree = grammar.parse(Path("test.mat").read_text())
print(parse_tree)


class Operator(Enum):
    PLUS = 0
    MINUS = 1
    MUL = 2
    DIV = 3


class Expr:
    pass


class Number(Expr):
    value: int

    def __init__(self, _value):
        self.value = _value


class BinOp(Expr):
    op: Operator
    left: Expr
    right: Expr

    def __init__(self, _left, _op, _right):
        self.op = _op
        self.left = _left
        self.right = _right


class CalculatorVisitor(NodeVisitor):
    def visit_mul_expr(self, node: Node, visited_children):
        # [numbers mul_ops] number
        if len(visited_children[0]) == 0:
            return visited_children[1]

        left_num = visited_children[0][0][0]
        curr_op = visited_children[0][0][1]
        for [num, op] in visited_children[0][1:]:
            left_num = BinOp(left_num, curr_op, num)
            curr_op = op

        return BinOp(left_num, curr_op, visited_children[1])

    def visit_add_expr(self, node: Node, visited_children):
        # [numbers mul_ops] number
        if len(visited_children[0]) == 0:
            return visited_children[1]

        left_num = visited_children[0][0][0]
        curr_op = visited_children[0][0][1]
        for [num, op] in visited_children[0][1:]:
            left_num = BinOp(left_num, curr_op, num)
            curr_op = op

        return BinOp(left_num, curr_op, visited_children[1])

    def visit_add_op(self, node: Node, visited_children):
        op_str = node.children[0].text
        match op_str:
            case "+":
                return Operator.PLUS
            case "-":
                return Operator.MINUS
            case _:
                raise Exception("Should not reach here")

    def visit_mul_op(self, node: Node, visited_children):
        op_str = node.children[0].text
        match op_str:
            case "*":
                return Operator.MUL
            case "/":
                return Operator.DIV
            case _:
                raise Exception("Should not reach here")

    def visit_number(self, node: Node, visited_children):
        value = int(node.text)
        return Number(value)

    def generic_visit(self, node: Node, visited_children):
        return visited_children


def interpret(expr: Expr):
    match expr:
        case Number(value=value):
            return value
        case BinOp(left=left, op=op, right=right):
            left_value = interpret(left)
            right_value = interpret(right)

            match op:
                case Operator.PLUS:
                    return left_value + right_value
                case Operator.MINUS:
                    return left_value - right_value
                case Operator.MUL:
                    return left_value * right_value
                case Operator.DIV:
                    return left_value / right_value
                case _:
                    raise Exception("Should not reach here")


iv = CalculatorVisitor()
ast = iv.visit(parse_tree)
print(interpret(ast))
