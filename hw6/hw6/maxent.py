import sys
import numpy as np
import math
import pandas as pd
from operator import itemgetter

class MaxEnt:

    def __init__(self, model_file):
        self.instances = []
        self.classes = []
        self.N = 0 #number of classes
        self.model = dict()
        self.sys_op = [] #system output lines
        self.read_model(model_file)
    
    def read_model(self, model_file):
        model_txt = [] #list of lines in model file
        cLabel = '' #current class label being read
        with open(model_file,'r') as f:
            model_str = f.read().strip()
            model_txt = model_str.split('\n')
        for line in model_txt:
            line = line.strip()
            if 'FEATURES FOR CLASS' in line:
                cIndex = len('FEATURES FOR CLASS')+1
                cLabel = line[cIndex:]
                self.classes.append(cLabel)
                self.model[cLabel] = dict()
            else:
                feature, val = line.split(' ')[0], float(line.split(' ')[1])
                self.model[cLabel][feature] = val
        self.N = len(self.classes)

    def test_evaluate(self,sys_output, test_file):
        confusion, accuracy = self.read_test_instances(test_file)

        with open(sys_output, 'w') as f:
            for line in self.sys_op:
                f.write(line)
                f.write('\n')

        self.evaluation(confusion, accuracy)
    
    def read_test_instances(self, instance_file):
        instances_list = [] # all raw instances
        self.instances = []
        inst_num = 0 #instance count
        confusion, accuracy = [[0]*self.N for i in range(self.N)], 0
        self.sys_op.append("%%%%% test data:\n")
        with open(instance_file) as f:
            inst_str = f.read().strip()
            instances_list = inst_str.split('\n')
        for line in instances_list:
            line = line.strip()
            if line == '':
                continue
            cLabel, raw_features_li = line.split()[0], line.split()[1:]
            #print('label',cLabel, 'feats',raw_features_li)
            features_di = dict() #feature counts for this instance 
            for feature in raw_features_li:
                #feature = feature.
                #if feature == ''
                f,val = feature.split(':')
                if f not in features_di.keys():
                    features_di[f] = val
                else:
                    features_di[f] += val
            predLabel, op_str = self.classify_inst(features_di)
            if predLabel == cLabel:
                accuracy += 1
            confusion[self.classes.index(cLabel)][self.classes.index(predLabel)] += 1
            self.sys_op.append('array:'+str(inst_num)+' '+op_str)
            inst_num += 1
        accuracy /= len(instances_list)
        return confusion, accuracy
    
    def classify_inst(self, features_di):
        results = dict()
        Z = 0
        for label in self.classes:
            val = self.model[label]["<default>"]
            for feat in features_di:
                if feat in self.model[label]:
                    val += self.model[label][feat]
            results[label] = math.exp(val)
            Z += results[label]
        for label in results:
            results[label] /= Z

        probs = list(results.items())
        probs = sorted(probs, key=itemgetter(1), reverse=True)
        op_str = probs[0][0] + " "
        for cLabel in probs:
            op_str += cLabel[0] + " " + str(round(cLabel[1], 5)) + " "
        return probs[0][0], op_str

    def evaluation(self, confusion, accuracy):
        op_str = "Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n             "
        for label in self.classes:
            op_str += label + " "
        op_str += "\n"
        for i in range(len(self.classes)):
            op_str += self.classes[i] + " "
            for num in  confusion[i]:
                op_str += str(num) + " "
            op_str += "\n"
        op_str += "\n"
        op_str += " Test accuracy=" + str(round(accuracy, 5)) + "\n\n\n"
        print(op_str)


            
        #self.model=pd.concat()

    #def calc_empirical_exp():
    #    self.N = self.training_instances
    #    for x_inst in self.training_instances:
    #        y=class_label
    #        for feat_t in x_inst:
    #            self.emp_exp[feat_t][y] += 1/self.N 
    #calculating P(y|x)
    


if __name__=="__main__":
    #test_data
    test_file = sys.argv[1]
    #model_file_q1
    model_file = sys.argv[2]
    #sys_output_file
    op_file = sys.argv[3]

    me = MaxEnt(model_file)
    me.test_evaluate(op_file, test_file)