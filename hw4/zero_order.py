from functions import getFunction
import sys
import math
import random
random.seed(10)

class ZeroOrder:
    def __init__(self, func_name, alpha, max_its, method_id, x1_val, x2_val):
        self.function = getFunction(func_name)
        self.alpha = alpha
        self.max_its = max_its
        self.method_id = method_id
        self.x1_val = x1_val
        self.x2_val = x2_val
        self.w = self.function([x1_val, x2_val])
        self.last_axis_flag = False


    def iterate(self):
        print(0, self.x1_val, self.x2_val, self.w)
        for k in range(1, max_its+1):

            if self.method_id == 1:
                optimal_x1, optimal_x2, optimal_w = self.random_search()
                if optimal_w < self.w:
                    self.x1_val = optimal_x1
                    self.x2_val = optimal_x2
                    self.w = optimal_w
                    print(k, self.x1_val, self.x2_val, self.w)
                else:
                    print(k, self.x1_val, self.x2_val, self.w)

            elif self.method_id == 2:
                optimal_x1, optimal_x2, optimal_w = self.coordinate_search()
                if optimal_w < self.w:
                    self.x1_val = optimal_x1
                    self.x2_val = optimal_x2
                    self.w = optimal_w
                    print(k, self.x1_val, self.x2_val, self.w)
                else:
                    print(k, self.x1_val, self.x2_val, self.w)
                    break


            elif self.method_id == 3:
                optimal_x1, optimal_x2, optimal_w = self.coordinate_descent(k)
                if optimal_w < self.w:
                    self.x1_val = optimal_x1
                    self.x2_val = optimal_x2
                    self.w = optimal_w
                    print(k, self.x1_val, self.x2_val, self.w)
                    self.last_axis_flag = False
                else:
                    print(k, self.x1_val, self.x2_val, self.w)
                    if not self.last_axis_flag:
                        self.last_axis_flag = True
                    else:
                        break

            else:
                raise ValueError("Wrong method_id. Try again!")

    def calculate(self, x1_dir_k, x2_dir_k):
        new_x1_val = self.x1_val + self.alpha * x1_dir_k
        new_x2_val = self.x2_val + self.alpha * x2_dir_k
        new_w = self.function([new_x1_val, new_x2_val])
        return new_w, new_x1_val, new_x2_val

    def find_optimal_dir(self, dir_list):
        optimal_x1 = None
        optimal_x2 = None
        optimal_w = math.inf
        for dir_k in dir_list:
            new_w, new_x1_val, new_x2_val = self.calculate(dir_k[0], dir_k[1])
            if new_w < optimal_w:
                optimal_x1 = new_x1_val
                optimal_x2 = new_x2_val
                optimal_w = new_w
        return optimal_x1, optimal_x2, optimal_w

    def random_search(self):
        dir_list = []
        for i in range(10):
            x1 = random.uniform(-1, 1)
            dir_k = math.sqrt(1 - math.pow(x1, 2))
            dir_list.append((x1, dir_k))
            dir_list.append((x1, -dir_k))
        return self.find_optimal_dir(dir_list)

    def coordinate_search(self):
        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return self.find_optimal_dir(dir_list)

    def coordinate_descent(self, k):
        if k % 2 == 0:
            dir_list = [(0, -1), (0, 1)]
        else:
            dir_list = [(-1, 0), (1, 0)]
        return self.find_optimal_dir(dir_list)

if __name__ == "__main__":
    func_name = sys.argv[1]
    alpha = float(sys.argv[2])
    max_its = int(sys.argv[3])
    method_id = int(sys.argv[4])
    x1_val = float(sys.argv[5])
    x2_val = float(sys.argv[6])

    z = ZeroOrder(func_name, alpha, max_its, method_id, x1_val, x2_val)
    z.iterate()