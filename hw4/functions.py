import math

func1 = lambda x: math.pow(x[0], 2) + math.pow(x[1], 2) + 2
func2 = lambda x: math.pow(x[0], 2) + (24 * math.pow(x[1], 2))
func3 = lambda x: math.pow(x[0], 2) + (120 * math.pow(x[1], 2))
func4 = lambda x: math.pow(x[0], 2) + (1200 * math.pow(x[1], 2))
g1 = lambda x: math.sin(3*x)
g2 = lambda x: math.sin(3*x) + (0.1 * math.pow(x, 2))
g3 = lambda x: math.pow(x, 2) + 0.2
g4 = lambda x: math.pow(x, 3)
g5 = lambda x: (math.pow(x, 4) + math.pow(x, 2) + (10 * x)) / 50
g6 = lambda x: math.pow(max(0, math.pow((3 * x) - 2.3, 3) + 1), 2) + math.pow(max(0, math.pow((-3 * x) + 0.7, 3) + 1), 2)


def getFunction(fstr):
	if fstr == "f1":
		return func1
	elif fstr == "f2":
		return func2
	elif fstr == "f3":
		return func3
	elif fstr == "f4":
		return func4
	elif fstr == "g1":
		return g1
	elif fstr == "g2":
		return g2
	elif fstr == "g3":
		return g3
	elif fstr == "g4":
		return g4
	elif fstr == "g5":
		return g5
	elif fstr == "g6":
		return g6
	else:
		raise ValueError("Wrong function name")
