import sys
from collections import defaultdict
import math
import re
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_count = defaultdict(int)
        self.joint_count = defaultdict(int)
        self.cond_prob = defaultdict(float)
        self.not_cond_prob = defaultdict(float)
        self.vocab = set()
        self.class_prob = defaultdict(float)
        self.total_count = 0

    def train_data(self, train_file, cond_prob_delta, class_prior_delta):
        f=open(train_file, 'r')
        text = f.read()
        lines = [line for line in text.split('\n') if line.strip() != '']
        self.total_count = len(lines)
        #first token is classname rest of it are features
        for line in lines:
            line = re.sub(':\d+', '', line)
            tokens = line.split()
            self.class_count[tokens[0]] += 1
            
            for i in range(1,len(tokens)):
                self.vocab.add(tokens[i])
                self.joint_count[(tokens[0],tokens[i])]+=1.0
        #print(self.joint_count)
        model_str = self.get_log_probs(cond_prob_delta, class_prior_delta)
        return model_str

    def read_data(self,test_file):
        f=open(test_file, 'r')
        text = f.read()
        lines = [line for line in text.split('\n') if line.strip() != '']
        instances = []
        #first token is classname rest of it are features
        for line in lines:
            line = re.sub(':\d+', '', line)
            tokens = line.split()
            label = tokens[0]
            features = tokens[1:]
            instances.append([label,features])
        return instances

    def get_log_probs(self, cond_prob_delta, class_prior_delta):
        model_str = []
        model_str.append('%%%%% prior prob P(c) %%%%%')
        for c in self.class_count:
            num = 1*class_prior_delta + self.class_count[c]
            denom = len(self.class_count.keys())*class_prior_delta + self.total_count #unique classes
            self.class_prob[c] = math.log(num/denom,10)
            instance_str = "%"+" "+str(c)+ " "+ str(num/denom)+" "+str(self.class_prob[c])
            model_str.append(instance_str)
        model_str.append('%%%%% conditional prob P(f|c) %%%%%')
        prev_c = ''
        c_instance_model = []
        for c,f in self.joint_count:
            num = 1*cond_prob_delta + self.joint_count[(c,f)]
            denom = 2*cond_prob_delta + self.class_count[c]
            
            prob = num/denom
            self.cond_prob[(c,f)] = math.log(prob,10)
            self.not_cond_prob[(c,f)] = math.log((1-prob),10) #1-P(w,k) log probability
            #self.z_log_prob[c] += self.not_cond_prob[(c,f)] 
            if c != prev_c:
                prev_c = c
                c_instance_model = sorted(c_instance_model)
                model_str.extend(c_instance_model)
                c_instance_model = []
                c_instance_model.append('%%%%% conditional prob P(f|c) c='+str(c)+' %%%%%')
            instance_str = str(f)+" "+str(c)+ " "+ str(prob)+" "+str(self.cond_prob[(c,f)])
            c_instance_model.append(instance_str)
            
        #model_str= sorted(model_str)
        return model_str
    
    def test_data(self, test_file):
        f=open(test_file, 'r')
        y,pred_y = [], []
        text = f.read()
        lines = [line for line in text.split('\n') if line.strip() != '']
        for line in lines:
            line = re.sub(':\d+', '', line)
            tokens = line.split()
            y.append(tokens[0])
            features = set(tokens[1:])
            pred_args.append(self.classify_doc(features))
        return y,pred_args
    
    def logsumexp(self,x):
        c = x.max()
        return c + np.log(np.sum(np.exp(x - c)))

    def classify_doc(self, test_features):
        #get probs for all classes
        args = []
        nb_probs = []
        classes = list(self.class_prob.keys())
        for c in self.class_prob:
            prior = self.class_prob[c]
            t1 = 0.0 #for f in test_features calculate log(1-P_joint)
            for f in self.vocab:
                if f in test_features:
                    t1 += self.cond_prob[(c,f)]
                else:
                    t1 += self.not_cond_prob[(c,f)]
                
            nb_probs.append(prior+t1)
        x = np.array(nb_probs)
        norm_class_prob = list(np.exp(x - self.logsumexp(x)))
        args = [(classes[i],norm_class_prob[i]) for i in range(len(x))]
        args.sort(key = lambda i:i[1], reverse = True)
        return args

    def evaluation(self, labels, train_confusion, train_accuracy, test_confusion, test_accuracy):
        eval_str = "Confusion matrix for the training data:\nrow is the truth, column is the system output\n\n             "
        for label in labels:
            eval_str += label + " "
        eval_str += "\n"
        for i in range(len(labels)):
            eval_str += labels[i] + " "
            for num in  train_confusion[i]:
                eval_str += str(num) + " "
            eval_str += "\n"
        eval_str += "\n"
        eval_str += " Training accuracy=" + str(train_accuracy) + "\n\n\n"

        eval_str += "Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n             "
        for label in labels:
            eval_str += label + " "
        eval_str += "\n"
        for i in range(len(labels)):
            eval_str += labels[i] + " "
            for num in  test_confusion[i]:
                eval_str += str(num) + " "
            eval_str += "\n"
        eval_str += "\n"
        eval_str += " Test accuracy=" + str(test_accuracy) + "\n\n\n"
        print(eval_str)

    def classify_train_test(self, train_file, test_file, sys_output):
        labels = list(self.class_count.keys())
        # [ [class class class] [] [] ] same features in order of list
        train_confusion = [[0]*len(labels) for i in range(len(labels))]
        test_confusion = [[0]*len(labels) for i in range(len(labels))]
        train_accuracy = 0
        test_accuracy = 0
        classify_str = "%%%%% training data:\n"
        # training vectors:
        i = 0
        train_instances = self.read_data(train_file)
        for vect in train_instances:
            
            true_label = vect[0]
            classify_str += "array:" + str(i) + " " + true_label+" "
            features = vect[1]
            sorted_count_classes = self.classify_doc(features)
            max_label = sorted_count_classes[0][0]
            max_count = sorted_count_classes[0][1]
            for label_count in sorted_count_classes:
                classify_str += label_count[0] + " " + str(label_count[1]) + " "
                if label_count[1] > max_count:
                    max_count = label_count[1]
                    max_label = label_count[0]
            classify_str += "\n"
            #print(labels,'true_label',true_label,'maxlabel',max_label)
            train_confusion[labels.index(true_label)][labels.index(max_label)] += 1
            if true_label == max_label:
                train_accuracy += 1
            i += 1
        classify_str += "\n\n%%%%% test data:\n"
        # test vectors:
        i = 0
        test_instances = self.read_data(test_file)
        for vect in test_instances:
            classify_str += "array:" + str(i)
            true_label = vect[0]
            features = vect[1]
            sorted_count_classes = self.classify_doc(features)
            max_label = sorted_count_classes[0][0]
            max_count = sorted_count_classes[0][1]
            for label_count in sorted_count_classes:
                classify_str += label_count[0] + " " + str(label_count[1]) + " "
                
            classify_str += "\n"
            #print(labels,'true_label',true_label,'maxlabel',max_label)
            test_confusion[labels.index(true_label)][labels.index(max_label)] += 1
            if true_label == max_label:
                test_accuracy += 1
            i += 1
        with open(sys_output, 'w') as f:
            f.write(classify_str)
        self.evaluation(labels, train_confusion, train_accuracy/len(train_instances), test_confusion, test_accuracy/len(test_instances))



if __name__=='__main__':
    # taking in parameters
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    class_prior_delta = float(sys.argv[3])
    cond_prob_delta = float(sys.argv[4])
    model_file = sys.argv[5]
    sys_output = sys.argv[6]
    nb = NaiveBayes()
    op_model = nb.train_data(train_file, cond_prob_delta, class_prior_delta)

    with open(model_file,'w+') as f:
        for line in op_model:
            f.write(line+"\n")
    
    nb.classify_train_test(train_file, test_file, sys_output)
