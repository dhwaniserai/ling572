import sys
from operator import itemgetter
import numpy as np
from math import exp, tanh


class SVM:
    def __init__(self):
        self.sv = []
        self.kernel = ''  # linear, polynomial, rbf, sigmoid
        self.gamma = 0
        self.coef = 0
        self.degree = 0
        self.rho = 0
        self.n_feat = 0

    def calc_inner_prod(self, v1, v2):
        #calculating inner prod acc to kernel type
        if self.kernel == 'linear':
            return np.inner(v1, v2)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.inner(v1, v2) + self.coef) ** self.degree
        elif self.kernel == 'rbf':
            return exp(-self.gamma * ((np.linalg.norm(v1-v2))**2))
        elif self.kernel == 'sigmoid':
            return tanh(self.gamma*np.inner(v1, v2) + self.coef)
        else:
            raise ValueError("Wrong kernel value. Check again!")

    def decode(self, data):
        #result is a list of tuples of type(true_label, pred_label, f(x))
        result = [] 
        for label, feat_vect in data:
            s = 0
            for weight, support_vect in self.sv:
                s += weight * self.calc_inner_prod(support_vect, feat_vect)
            s = s - self.rho
            if s >= 0:
                result.append((label, 0, s))
            else:
                result.append((label, 1, s))

        return result


    def read_model(self, model_file):
        #reading svm model params and support vectors
        model_lines = []
        with open(model_file, 'r') as svm_file:
            model_lines = svm_file.readlines()
        
        param_index = model_lines.index('SV\n')
        for i in range(param_index):
            #parameters
            line = model_lines[i].strip()
            if line != '':
                params = line.split(' ')
                if params[0] == 'kernel_type':
                    self.kernel=params[1]
                elif params[0] == 'gamma':
                    self.gamma=float(params[1])
                elif params[0] == 'coef0':
                    self.coef=float(params[1])
                elif params[0] == 'degree':
                    self.degree=float(params[1])
                elif params[0] == 'rho':
                    self.rho=float(params[1])
        
        #support vectors
        sv = []
        max_feat_index = 0
        for i in range(param_index+1,len(model_lines)):
            line = model_lines[i].strip()
            if line != '':
                seq = line.strip(' ').split(' ')
                weight = float(seq[0])
                support_vect = {}
                for feat in seq[1:]:
                    feat_index, feat_val = int(feat.split(':')[0]), float(feat.split(':')[1])
                    support_vect[feat_index] = feat_val
                max_feat_index = max(max_feat_index, max(support_vect.keys()))
                sv.append((weight, support_vect))
                

        self.n_feat=max_feat_index + 1

        full_sv = []
        for weight, vector in sv:
            for i in range(0, self.n_feat):
                if i not in vector:
                    vector[i] = 0
            sorted_sv = np.array(sorted(vector.items(), key=itemgetter(0)))
            full_sv.append((weight, sorted_sv.transpose()[1]))
        self.sv=full_sv
        


def read_data(filename, model_features):
    #read test data acc to number of features in the model
    data = []
    test_file_lines = []
    with open(filename, 'r') as f:
        test_file_lines = f.readlines()
    for line in test_file_lines:
        line = line.strip()
        if line != '':
            seq = line.split(' ')
            true_label = int(seq[0])  #gold label
            feat_vect = {} #feature vector
            for feat in seq[1:]:
                feat_index, feat_val = int(feat.split(':')[0]), float(feat.split(':')[1])
                feat_vect[feat_index] = feat_val
            for i in range(0, model_features):
                if i not in feat_vect:
                    feat_vect[i] = 0
            sorted_sv = np.array(sorted(feat_vect.items(), key=itemgetter(0)))[:model_features]
            data.append((true_label, sorted_sv.transpose()[1]))

    return data


def write_output(result, op_file):
    correct, total = 0.0, 0.0
    with open(op_file, 'w') as f:
        for true, pred, val in result:
            out_str = str(true) + ' ' + str(pred) + ' ' + str(round(val, 5)) + '\n'
            f.write(out_str)
            total += 1.0
            if true == pred:
                correct += 1.0
    print("Test accuracy =", str(correct/total*100))


if __name__ == "__main__":

    test_file = sys.argv[1]
    model_file = sys.argv[2]
    sys_output = sys.argv[3]

    svm_model = SVM()
    svm_model.read_model(model_file)
    test_data = read_data(test_file, svm_model.n_feat)
    result = svm_model.decode(test_data)
    write_output(result, sys_output)