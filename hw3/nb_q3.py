import sys
import numpy as np
from math import log2
from collections import Counter, defaultdict
from math import log10
from operator import itemgetter
from decimal import Decimal
import time


class MultinomialNaiveBayes():
    def __init__(self, prior_delta, prob_delta):
        self.class_prior_delta = prior_delta
        self.cond_prob_delta = prob_delta
        self.vocab = None
        self.train_data = None
        self.class_count = None
        self.joint_word_class_count = None

        self.prob_class = {}
        self.p_joint_word_class = {}
    
    def read_data(self, filename, test=False):
        with open(filename, 'r') as f:
            sentences = f.readlines()
        
        vocab = set()
        y = []
        X = []
        wc = []
        for instance in sentences:
            data = instance.strip("\n ").split()
            label = data[0]
            y.append(label)
            features = defaultdict(int)
            for feature in data[1:]:
                word, count = feature.strip().split(":")
                features[word] += int(count)
                vocab.add(word)
            X.append(features)

        
        data = np.array([X, y])

        if test:
            return data

        self.vocab = vocab
        self.class_count = dict(Counter(y))
        self.train_data = data
    
    def train(self, model_out):
        label_set = set(self.train_data[-1])
        prob_class = {}  
        n_labels = len(self.class_count)
        n_instances = self.train_data.shape[1]
        n_vocab = len(self.vocab)

        n_class = defaultdict(int)
        n_wordclass = defaultdict(int)

        for label, count in self.class_count.items():
            prob_class[label] = (self.class_prior_delta + count) / (n_labels*self.class_prior_delta + n_instances)
            

        for features, label in self.train_data.transpose():
            for word, count in features.items():
                n_class[label] += count
                n_wordclass[(label, word)] += count


        p_joint_word_class = defaultdict(float)  
        extra = len(self.vocab)*self.cond_prob_delta

        for label in self.class_count:
            for word in self.vocab:
                if (label, word) not in n_wordclass:
                    p_joint_word_class[(label, word)] = self.cond_prob_delta / (extra + n_class[label])
                else:
                    p_joint_word_class[(label, word)] += ((self.cond_prob_delta + n_wordclass[(label, word)]) / (extra + n_class[label]))

        self.prob_class = prob_class
        self.p_joint_word_class = p_joint_word_class

        self.print_model(model_out)
    
    def print_model(self, filename):
        model_str = f"%%%%% prior prob P(c) %%%%%\n"

        for label, prob in self.prob_class.items():
            model_str += f"{label}\t{prob}\t{log10(prob)}\n"
        
        model_str += f"%%%%% conditional prob P(f|c) %%%%%\n"

        for label in self.class_count:
            model_str += f"%%%%% conditional prob P(f|c) c={label} %%%%%\n"

            for word in sorted(self.vocab):
                prob = self.p_joint_word_class[(label, word)]
                model_str += f"{word}\t{label}\t{prob}\t{log10(prob)}\n"
        
        with open(filename, "w") as f:
            f.write(model_str)
    
    def test(self, test_data_file, sys_out_file, datatype="test"):
        # print(test_data_file, sys_out_file)
        data = self.read_data(test_data_file, test=True)
        labels = sorted(self.class_count.keys())
        conf_matrix = np.zeros((len(labels), len(labels)))
        sys_str = f"\n%%%%% {datatype} data:"

        for idx, features in enumerate(data[0]):
            log_prob_wclass = {key: log10(value) for key,value in self.prob_class.items()}
            prob_classx = defaultdict(float)
            
            for label in labels:
                for word in self.vocab:
                    if word in features:
                        count = features[word]
                        log_prob_wclass[label] += log10(self.p_joint_word_class[(label, word)])*count - sum([log10(num) for num in range(1, count+1)])
            
            smooth = min(log_prob_wclass.values())
            sig_pxc_pc = sum([(10**Decimal(log_prob_wclass[l]-smooth)) for l in labels])

            for label in labels:
                prob_classx[label] = (10**Decimal(log_prob_wclass[label]-smooth))/sig_pxc_pc
            
            results = sorted(prob_classx.items(), key=itemgetter(1), reverse=True)
            conf_matrix[labels.index(results[0][0])][labels.index(data[1][idx])] += 1
            sys_str += f"\narray:{idx}\t{data[1][idx]}"
            for ci, pi in results:
                sys_str += f"\t{ci}\t{pi}"
        self.evaluation(conf_matrix, labels, datatype)
        with open(sys_out_file, "a") as f:
            f.write(sys_str)

    def evaluation(self, conf_matrix, labels, datatype):
        acc_str = f"Confusion matrix for the {datatype} data:\n"
        acc_str += f"row is the truth, column is the system output\n\n"
        acc_str += "\t"+"\t".join(labels)
        for idx, label in enumerate(labels):
            acc_str += f"\n{label}\t"+"\t".join(map(str, conf_matrix[idx]))
        
        accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
        
        acc_str += f"\n\n{datatype} accuracy={accuracy}\n\n"

        print(acc_str)





train_file = sys.argv[1]
test_file = sys.argv[2]
class_prior_delta = float(sys.argv[3])
cond_prob_delta = float(sys.argv[4])
model_file = sys.argv[5]
sys_output = sys.argv[6]

mnb = MultinomialNaiveBayes(class_prior_delta, cond_prob_delta)
mnb.read_data(train_file)
mnb.train(model_file)

with open(sys_output, "w") as f:
    f.write("")
mnb.test(train_file, sys_output, datatype="train")
mnb.test(test_file, sys_output)