from formula import Var, Not, And, Or, Xor, sat


x = Var("x")
y = Var("y")

print("basic term printing")
print(x)
print(y)
print(Not(x))
print(And(x, y))
print(Or(x, y))
print(Xor(x, y))
print()

print("operator overloads; should print out the same thing as above")
print(x)
print(y)
print(~x)
print(x & y)
print(x | y)
print(x ^ y)
print()

print("conversion to NANDs; should print only NANDs")
print(Not(x).to_nands())
print(And(x, y).to_nands())
print(Or(x, y).to_nands())
print(Xor(x, y).to_nands())
print()

print("solutions")
print(Xor(x, y).solve())  # SAT
print(And(Xor(x, y), y).solve())  # SAT
print(And(And(Xor(x, y), y), x).solve())  # UNSAT (None)
print()

# Now for the actual DSL
print("the sat function")
print(sat(lambda x: x & ~x))  # UNSAT (None)
print(sat(lambda x, y: (x & y) ^ (y & x)))  # UNSAT (None)
print(sat(lambda x, y, z: (x & (y | z)) ^ ((x & y) | (x & z))))  # UNSAT (None)
print(sat(lambda x, y, z: (y & (y | z)) ^ ((x & y) | (x & z))))  # SAT
print(sat(lambda x, y, z: (x & (y ^ z)) ^ ((x & y) ^ (x & z))))  # UNSAT (None)
print(sat(lambda x, y, z: (x | (y ^ z)) ^ ((x | y) ^ (x | z))))  # SAT
