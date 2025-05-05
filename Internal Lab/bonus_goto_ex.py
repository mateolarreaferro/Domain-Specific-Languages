from bonus_goto import goto, label

# Example 1

for i in range(1, 10):
    for j in range(1, 10):
        for k in range(1, 10):
            print(i, j, k)
            if k == 3:
                goto.end

label.end

# Example 2

label.start
for i in range(1, 4):
    print(i)
    if i == 2:
        try:
            output = message
        except NameError:
            print("Oops - forgot to define 'message'!  Start again.")
            message = "Hello world"
            goto.start
print(output)

# Example 3: Maze
# prints: hi

label.maze_start
goto.m6
label.m0
print("a", end="")
label.m1
goto.maze_end
print("b", end="")
label.m2
goto.m5
print("c", end="")
label.m3
goto.m9
print("d", end="")
label.m4
goto.m2
print("e", end="")
label.m5
goto.m3
print("f", end="")
label.m6
print("h", end="")
goto.m4
label.m7
print("i", end="")
goto.m1
label.m8
print("j", end="")
label.m9
goto.m7
label.maze_end
print()
