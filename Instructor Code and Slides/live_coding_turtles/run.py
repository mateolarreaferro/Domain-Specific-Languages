from turtle import Turtle
from random import random
import sys

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from parsimonious.nodes import Node
from pathlib import Path
from enum import Enum

grammar = Grammar(Path("grammar.peg").read_text())
tree = grammar.parse(Path(sys.argv[1]).read_text())


class Expr:
    pass


class Stmt:
    pass


class TurtleE(Expr):
    pass


class Number(Expr):
    value: int

    def __init__(self, _value):
        self.value = _value


class String(Expr):
    value: str

    def __init__(self, _value):
        self.value = _value


class Array(Expr):
    ty: Expr
    count: int

    def __init__(self, _ty, _count):
        self.ty = _ty
        self.count = _count


class Variable(Expr):
    name: str

    def __init__(self, _name):
        self.name = _name


class FunctionCall(Expr):
    name: str
    args: list[Expr]

    def __init__(self, _name, _args):
        self.name = _name
        self.args = _args


class Operator(Enum):
    PLUS = 0
    MINUS = 1
    TIMES = 2
    DIVIDE = 3


class BinOp(Expr):
    left: Expr
    right: Expr
    op: Operator

    def __init__(self, _left, _op, _right):
        self.left = _left
        self.op = _op
        self.right = _right

    def __repr__(self):
        return f"({self.left}) {self.op} ({self.right})"


class Block(Stmt):
    stmts: list[Stmt]

    def __init__(self, _stmts):
        self.stmts = _stmts


class Let(Stmt):
    name: str
    value: Expr

    def __init__(self, _name, _value):
        self.name = _name
        self.value = _value


class Ask(Stmt):
    name: str
    block: Block

    def __init__(self, _name, _block):
        self.name = _name
        self.block = _block


class OnTick(Stmt):
    name: str
    block: Block

    def __init__(self, _name, _block):
        self.name = _name
        self.block = _block


class TurtleVisitor(NodeVisitor):
    def visit_program(self, node, visited_children):
        return Block(list([v[0] for v in visited_children]))

    def visit_ontick_block(self, node, visited_children):
        return OnTick(visited_children[1], Block(visited_children[3]))

    def visit_ask_block(self, node, visited_children):
        return Ask(visited_children[1], Block(visited_children[3]))

    def visit_statement(self, node, visited_children):
        return visited_children[0][0]

    def visit_binding(self, node, visited_children):
        return Let(visited_children[0], visited_children[2])

    def visit_expr(self, node, visited_children):
        return visited_children[0]

    def visit_unary_expr(self, node, visited_children):
        return visited_children[0]

    def visit_add_expr(self, node, visited_children):
        # structure of nodes:
        #  [[list of (number op) pairs], number]
        if len(visited_children[0]) == 0:
            return visited_children[1]

        left_num = visited_children[0][0][0]
        curr_op = visited_children[0][0][1]
        for [num, op] in visited_children[0][1:]:
            left_num = BinOp(left_num, curr_op, num)
            curr_op = op

        return BinOp(left_num, curr_op, visited_children[1])

    def visit_add_op(self, node, visited_children):
        op_str = node.children[0].text
        match op_str:
            case "+":
                return Operator.PLUS
            case "-":
                return Operator.MINUS
            case _:
                raise Exception("Invalid add operator")

    def visit_mul_expr(self, node, visited_children):
        # structure of nodes:
        #  [[list of (number op) pairs], number]
        if len(visited_children[0]) == 0:
            return visited_children[1]

        # [[5, *], [2, /], [4, *]], 5
        # (((5 * 2) / 4) * 5)
        left_num = visited_children[0][0][0]
        curr_op = visited_children[0][0][1]
        for [num, op] in visited_children[0][1:]:
            left_num = BinOp(left_num, curr_op, num)
            curr_op = op

        return BinOp(left_num, curr_op, visited_children[1])

    def visit_mul_op(self, node, visited_children):
        op_str = node.children[0].text
        match op_str:
            case "*":
                return Operator.TIMES
            case "/":
                return Operator.DIVIDE
            case _:
                raise Exception("Invalid mul operator")

    def visit_function_call(self, node, visited_children):
        name, _, arg_list, _ = visited_children
        if len(arg_list) == 0:
            return FunctionCall(name, [])
        else:
            return FunctionCall(name, arg_list[0])

    def visit_arg_list(self, node, visited_children):
        if len(visited_children[0]) == 0:
            return [visited_children[1]]

        args = []
        for [arg, comma] in visited_children[0]:
            args.append(arg)
        args.append(visited_children[1])
        return args

    def visit_variable(self, node, visited_children):
        return Variable(visited_children[0])

    def visit_name(self, node, visited_children):
        return node.children[0].text

    def visit_array(self, node, visited_children):
        return Array(visited_children[0], visited_children[2].value)

    def visit_turtle(self, node, visited_children):
        return TurtleE()

    def visit_literal(self, node, visited_children):
        return visited_children[0]

    def visit_string(self, node, visited_children):
        text = node.children[1].text
        return String(text)

    def visit_name(self, node, visited_children):
        return node.children[0].text

    def visit_number(self, node, visited_children):
        value = int(node.text)
        return Number(value)

    def generic_visit(self, node, visited_children):
        return visited_children


def interpret_expr(expr, bindings, context):
    match expr:
        case TurtleE():
            return Turtle()
        case Number(value=value):
            return value
        case String(value=value):
            return value
        case Array(ty=ty, count=count):
            return [interpret_expr(ty, bindings, context) for i in range(0, count)]
        case Variable(name=name):
            return bindings[name]
        case BinOp(left=left, right=right, op=op):
            left_val = interpret_expr(left, bindings, context)
            right_val = interpret_expr(right, bindings, context)

            match op:
                case Operator.PLUS:
                    return left_val + right_val
                case Operator.MINUS:
                    return left_val - right_val
                case Operator.TIMES:
                    return left_val * right_val
                case Operator.DIVIDE:
                    return left_val / right_val

        case FunctionCall(name=name, args=args):
            arg_values = [interpret_expr(arg, bindings, context) for arg in args]
            match name:
                case "forward":
                    context.fd(arg_values[0])
                case "right":
                    context.rt(arg_values[0])
                case "heading":
                    context.setheading(arg_values[0])
                case "position":
                    context.setposition(arg_values[0], arg_values[1])
                case "pendown":
                    context.pendown()
                case "penup":
                    context.penup()
                case "random":
                    return random() * arg_values[0]
                case "print":
                    print(arg_values[0])
                    return None

        case _:
            print(f"UNKNOWN EXPR: {expr}")


def interpret_stmt(stmt, bindings, context):
    match stmt:
        case Let(name=name, value=expr):
            value = interpret_expr(expr, bindings, context)

            if context is not None:
                match name:
                    case "shape":
                        context.shape(value)
                    case "color":
                        context.color(value)
                    case _:
                        bindings[name] = value
            else:
                bindings[name] = value
        case Ask(name=name, block=block):
            turtles = bindings[name]
            match turtles:
                case Turtle():
                    interpret_block(block, bindings, turtles)
                case list():
                    for turtle in turtles:
                        interpret_block(block, bindings, turtle)
        case OnTick(name=name, block=block):
            turtles = bindings[name]
            while True:
                match turtles:
                    case Turtle():
                        interpret_block(block, bindings, turtles)
                    case list():
                        for turtle in turtles:
                            interpret_block(block, bindings, turtle)
        case _:
            interpret_expr(stmt, bindings, context)


def interpret_block(block, bindings, context):
    for stmt in block.stmts:
        interpret_stmt(stmt, bindings, context)


class Type:
    pass


class BasicType(Enum):
    NUMBER = 0
    TURTLE = 1
    STRING = 2


class TyArray(Type):
    inner: Type
    count: int

    def __init__(self, _inner, _count):
        self.inner = _inner
        self.count = _count

    def __eq__(self, other):
        return self.inner == other.inner and self.count == other.count


class TyFunction(Type):
    params: list[Type]
    ret: Type

    def __init__(self, _params, _ret):
        self.params = _params
        self.ret = _ret


def type_block(block, bindings, context):
    for stmt in block.stmts:
        type_stmt(stmt, bindings, context)


builtin_properties = {
    "shape": BasicType.STRING,
    "color": BasicType.STRING,
}


def type_stmt(stmt, bindings, context):
    match stmt:
        case Let(name=name, value=expr):
            ty = type_expr(expr, bindings, context)

            # todo
            if context is not None and name in builtin_properties:
                assert ty == builtin_properties[name]
            else:
                bindings[name] = ty
        case Ask(name=name, block=block):
            turtle = bindings[name]
            type_block(block, bindings, turtle)
        case OnTick(name=name, block=block):
            turtle = bindings[name]
            type_block(block, bindings, turtle)
        case _:
            type_expr(stmt, bindings, context)


builtin_functions = {
    "forward": TyFunction([BasicType.NUMBER], None),
    "right": TyFunction([BasicType.NUMBER], None),
    "heading": TyFunction([BasicType.NUMBER], None),
    "position": TyFunction([BasicType.NUMBER, BasicType.NUMBER], None),
    "random": TyFunction([BasicType.NUMBER], BasicType.NUMBER),
    "pendown": TyFunction([], None),
    "penup": TyFunction([], None),
    "print": TyFunction([BasicType.NUMBER], None),
}


def type_expr(expr, bindings, context):
    match expr:
        case TurtleE():
            return BasicType.TURTLE
        case Number(value=value):
            return BasicType.NUMBER
        case String(value=value):
            return BasicType.STRING
        case Array(ty=ty, count=count):
            return TyArray(ty, count)
        case Variable(name=name):
            return bindings[name]
        case BinOp(left=left, right=right, op=op):
            left_ty = type_expr(left, bindings, context)
            right_ty = type_expr(right, bindings, context)

            assert left_ty == BasicType.NUMBER and right_ty == BasicType.NUMBER
            return BasicType.NUMBER
        case FunctionCall(name=name, args=args):
            arg_tys = [type_expr(arg, bindings, context) for arg in args]
            function_ty = builtin_functions[name]

            assert len(function_ty.params) == len(
                arg_tys
            ), f"{name} has {len(function_ty.params)} params, but got {len(arg_tys)} args in {expr}"
            for param, arg in zip(function_ty.params, arg_tys):
                assert param == arg, f"{param} != {arg} in {expr}"

            return function_ty.ret
        case _:
            print(f"UNKNOWN: {expr}")


v = TurtleVisitor()
ast = v.visit(tree)
bindings = {}
type_block(ast, bindings, None)
interpret_block(ast, bindings, None)
