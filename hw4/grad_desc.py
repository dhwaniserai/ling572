import sys
import math
import numdifftools as nd

from functions import getFunction

class GradientDescent:

    def __init__(self, func_name, alpha, max_its, x1_val, x2_val=None):
        self.function = getFunction(func_name)
        self.alpha = alpha
        self.max_its = max_its
        self.x1_val = x1_val
        self.x2_val = x2_val # check for x2
        if self.x2_val == None:
            self.w = self.function(x1_val)
        else:
            self.w = self.function([x1_val, x2_val])
        self.m = 5
        self.epsilon = 0.001
        self.converge = True # check convergence


    def iterate(self):
        grad = nd.Gradient(self.function)
        if self.x2_val == None:
            print(0, self.x1_val, self.w)
        else:
            print(0, self.x1_val, self.x2_val, self.w)

        for k in range(1, self.max_its+1):
            if self.x2_val == None:
                x1_grad = grad([self.x1_val])
                new_x1_val = self.x1_val - (self.alpha * x1_grad)
                new_w = self.function(new_x1_val)
                if k > (self.max_its - self.m):
                    if (abs(self.x1_val - new_x1_val) >= self.epsilon) or (abs(self.w - new_w) >= self.epsilon):
                        self.converge = False
                self.x1_val = new_x1_val
                self.w = new_w
                print(k, self.x1_val, self.w)
            else:
                cur_grad = grad([self.x1_val, self.x2_val])
                new_x1_val = self.x1_val - (self.alpha * cur_grad[0])
                new_x2_val = self.x2_val - (self.alpha * cur_grad[1])
                new_w = self.function([new_x1_val, new_x2_val])
                if k > (self.max_its - self.m):
                    last = math.sqrt(math.pow(self.x1_val - new_x1_val, 2) + math.pow(self.x2_val - new_x2_val, 2))
                    if (abs(last) >= self.epsilon) or abs(self.w - new_w) >= self.epsilon:
                        self.converge = False
                self.x1_val = new_x1_val
                self.x2_val = new_x2_val
                self.w = new_w
                print(k, self.x1_val, self.x2_val, self.w)
        if not self.converge:
            print("no")
        else:
            if self.x2_val == None:
                if ((abs(self.x1_val) < math.pow(10, 8)) and (abs(self.w) < math.pow(10, 8))):
                    print("yes")
                else:
                    print("yes-but-diverge")
            else:
                val = math.sqrt(math.pow(self.x1_val, 2) + math.pow(self.x2_val, 2))
                if ((abs(val) < math.pow(10, 8)) and (abs(self.w) < math.pow(10, 8))):
                    print("yes")
                else:
                    print("yes-but-diverge")

#q3
if __name__ == "__main__":
    func_str = sys.argv[1]
    alpha = float(sys.argv[2])
    max_its = int(sys.argv[3])
    x1_val = float(sys.argv[4])
    x2_val = None
    if len(sys.argv) > 5:
        x2_val = float(sys.argv[5])

    z = GradientDescent(func_str, alpha, max_its, x1_val, x2_val)
    z.iterate()