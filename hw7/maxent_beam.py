import sys
import numpy as np
import math
import pandas as pd
from operator import itemgetter

class Node:

    def __init__(self, sent_ind, word_ind, inst_word, pos, features, pred_label='', cur_prob=0, path_prob=0.0, prev_node=None):
        self.sent_ind = sent_ind
        self.word_ind = word_ind
        self.name = inst_word
        self.pos = pos
        self.features = features
        self.pred_label = pred_label
        self.cur_prob = cur_prob
        self.path_prob = path_prob
        self.prev_node = prev_node
    
def read_model(model_file):
    model_txt = [] #list of lines in model file
    cLabel = '' #current class label being read
    classes = list()
    model = dict()
    with open(model_file,'r') as f:
        model_str = f.read().strip()
        model_txt = model_str.split('\n')
    for line in model_txt:
        line = line.strip()
        if line == '':
            continue
        elif 'FEATURES FOR CLASS' in line:
            cIndex = len('FEATURES FOR CLASS')+1
            cLabel = line[cIndex:]
            classes.append(cLabel)
            model[cLabel] = dict()
        else:
            feature, val = line.split(' ')
            model[cLabel][feature] = float(val)
    
    return model,classes

def read_boundary(boundary_file):
    line_boundaries = list()
    with open(boundary_file,'r') as f:
        b_text = f.read().strip()
    for line in b_text.split('\n'):
        line = line.strip()
        if line != '':
            line_boundaries.append(int(line))
    return line_boundaries

def read_test_data(data_file, data_boundaries):
    instances = []
    data_lines = []
    with open(data_file, 'r') as f:
        data_str = f.read().strip()
        data_lines = data_str.split('\n')
    for line in data_lines:
        line = line.strip()
        if line != '':
            feats = line.split(' ')
            sent_ind, word_ind, word = feats[0].split('-', 2)
            label = feats[1]
            data_point = Node(int(sent_ind)-1, int(word_ind), word, label, set(feats[2::2]))
            instances.append(data_point)
    return instances

def generate_prev_features(word_ind, prev_node):
    if word_ind==0:
        return ('prevT=BOS', 'prevTwoTags=BOS+BOS')
    elif word_ind==1:
        p1_tag = 'prevT='+prev_node.pred_label
        p2_tag = 'prevTwoTags=BOS+'+prev_node.pred_label
        return (p1_tag,p2_tag)
    else:
        p1_tag = 'prevT='+prev_node.pred_label
        p2_tag = 'prevTwoTags='+prev_node.pred_label + '+' +prev_node.prev_node.pred_label
        return (p1_tag,p2_tag)

'''
def test_evaluate(self,sys_output, test_file):
    confusion, accuracy = self.read_test_instances(test_file)

    with open(sys_output, 'w') as f:
        for line in self.sys_op:
            f.write(line)
            f.write('\n')

    self.evaluation(confusion, accuracy)
'''

def evaluate_test_instances(instances_data, model, line_boundaries, classes, topN, topK, beam_size):
    
    prev_nodes = []
    inst_num, word_index = 0, 0 #instance count, current word index
    accuracy = 0
    sys_op = []
    sys_op.append("%%%%% test data:\n")
    for data_point in instances_data:
        word_ind = data_point.word_ind
        #print('word_ind', word_ind)
        if word_ind==0:
            prev_nodes.clear()
            features = data_point.features
            features.union(generate_prev_features(word_ind, None))
            results = calculate_topn(model, classes, features, topN)
            #print('0 results', results)
            for cLabel,prob in results:
                new_point = Node(data_point.sent_ind, data_point.word_ind, data_point.name, data_point.pos, data_point.features, cLabel, \
                    prob, math.log10(prob), None)
                prev_nodes.append(new_point)
                #print('prev nodes', prev_nodes)
        else:
            results=[]
            #print('inside else')
            for last_node in prev_nodes:
                features = data_point.features
                features.union(generate_prev_features(word_ind, last_node))
                cur_result = calculate_topn(model,classes,features,topN)
                #print('cur_result', cur_result)
                for cLabel, prob in cur_result:
                    results.append((cLabel, prob, math.log10(prob)+last_node.path_prob, last_node))
            
            surviving_results = prune(results, topK, beam_size)
            prev_nodes.clear()
            for cLabel, cur_prob, path_prob, last_node in surviving_results:
                # Create a new set of nodes
                new_point = Node(data_point.sent_ind, word_ind, data_point.name, data_point.pos, data_point.features,
                                cLabel, cur_prob, path_prob, last_node)
                prev_nodes.append(new_point)

        sent_ind = data_point.sent_ind
        #print('bounds ', line_boundaries[5526:])
        if word_ind>= line_boundaries[sent_ind]-1: 
            #last word
            current = prev_nodes[0]
            op_str_list = []
            while current != None:
                op_str = str(current.sent_ind+1)+'-'+str(current.word_ind)+'-'+current.name + ' ' + current.pos + ' ' + current.pred_label + ' ' + \
                         str(round(current.cur_prob,5))
                op_str_list.append(op_str)
                if current.pred_label==current.pos:
                    accuracy += 1
                current = current.prev_node
            op_str_list.reverse()
            sys_op.extend(op_str_list)
            

    accuracy /= len(instances_data)
    print("Accuracy = ", accuracy)
    return sys_op

def calculate_topn(model, classes, features, topN):
    results = dict()
    for cLabel in classes:
        sum_exp = model[cLabel]['<default>']
        for feat in features:
            if cLabel in model.keys():
                if feat in model[cLabel]:
                    sum_exp += model[cLabel][feat]
        results[cLabel] = math.exp(sum_exp)

    z = sum(results.values())
    for cLabel in classes:
        results[cLabel] = results[cLabel] / z
    sorted_results = sorted(results.items(), key=itemgetter(1), reverse=True)  # sorting by probability
    #print('calc results', sorted_results)
    return sorted_results[:topN]

def prune(results, top_k, beam_size):
    
    #print('results', results)
    sorted_result = sorted(results, key=itemgetter(2), reverse=True)[:top_k] #sorting in descending order
    #print('sorted', sorted_result)
    max_prob = sorted_result[0][2]
    surviving_nodes = []
    for cLabel, cur_prob, path_prob, last_node in sorted_result:
        if path_prob + beam_size >= max_prob:
            surviving_nodes.append((cLabel, cur_prob, path_prob, last_node))
    return surviving_nodes


def main():
    #test_data
    test_file = sys.argv[1]
    #boundary_file
    boundary_file = sys.argv[2]
    #model_file
    model_file = sys.argv[3]
    #sys_output_file
    op_file = sys.argv[4]
    #beam_size
    beam_size = int(sys.argv[5])
    #topN
    topN = int(sys.argv[6])
    #topK
    topK = int(sys.argv[7])

    beam_model, classes = read_model(model_file)
    line_boundaries = read_boundary(boundary_file)
    test_data = read_test_data(test_file, line_boundaries)
    sys_op = evaluate_test_instances(test_data, beam_model, line_boundaries, classes, topN, topK, beam_size)

    with open(op_file,'w') as f:
        for line in sys_op:
            f.write(line+'\n')
#main file
main()