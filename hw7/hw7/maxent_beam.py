import sys
import math
import numpy as np
import time

def read_data(data_file):
    data_lines = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            features = {}
            for x in range(2, len(line),2):
                features[line[x]] = line[x+1] 
            data_lines.append((line[0], line[1], features))
    return data_lines

def read_model(model_file):
    model, features, clabels, all_feats = {}, set(), set(), set()
    currentClass = ''
    with open(model_file, 'r') as f:
        for l in f:
            l = l.strip().split(' ')
            if len(l) > 2: 
                currentClass = l[-1]
                clabels.add(l[-1])
                model[currentClass] = {}
            else:
                features.add(l[0])
                all_feats.add(l[0])
                model[currentClass][l[0]] = float(l[1])
    return [model, features, clabels, ], len(all_feats)

def read_boundary(boundary_file):
    boundaries = list()
    with open(boundary_file,'r') as f:
        b_text = f.read().strip()
    for line in b_text.split('\n'):
        line = line.strip()
        if line != '':
            boundaries.append(int(line))
    return boundaries

def calculate_p(features, model):
    results = dict()
    
    for label in model[2]:
        sum_exp = model[0][label]['<default>']
        for feature in features:
            if feature in model[0][label]: 
                sum_exp += model[0][label][feature]
        results[label] = math.exp(sum_exp)
    z = sum(results.values())
    for label in model[2]:
        results[label] /= z
    return results

def prune(beam_size, topK, instances):
    max_prob = max(instances.values())[0]
    surviving_nodes = {}
    for tag in instances: #new potential nodes
        if (math.log(instances[tag][0]) + beam_size) >= math.log(max_prob):
            surviving_nodes[tag] =  instances[tag]
    new={}
    for cur_path in sorted(surviving_nodes.items(), key = lambda x: x[1][0], reverse=True)[:topK]:#select topK highest prob 
        new[cur_path[0]] = cur_path[1]
    return new

def beam_decoder(sentence, instances):
    sent_len = len(instances) 
    candidate = max(instances[sent_len-1])
    cur_path = [candidate]
    output = []
    for i in range(sent_len):
        probs = instances[sent_len-i-1][candidate][0]
        gold = sentence[sent_len-i-1][1]
        instance = sentence[sent_len-i-1][0]
        output.append("{} {} {} {}\n".format(instance, gold, candidate, probs))
        candidate = instances[sent_len-i-1][candidate][1]
        cur_path.append(candidate)
    op_str = ''
    for i in reversed(output[1:]): #Remove last
        op_str += i
    return list(reversed(cur_path))[1:], op_str # remove BOS tag from path

def write_output(candidate, true_gold, fnum):
    clabels = set(true_gold).union(set(candidate))
    d = len(clabels)
    print("class_num={} feat_num={}\n".format(d, fnum))
    print("Confusion matrix for the test data:\nrow is the truth, column is the system output\n")
    cand_len = len(candidate)
    m = np.zeros([d,d])
    label2idx, idx2label = {}, {}
    index, count = 0, 0
    for label in clabels:
        label2idx[label] = index
        idx2label[index] = label
        index += 1
    for j in range(cand_len):
        m[label2idx[candidate[j]]][label2idx[true_gold[j]]] += 1
        if candidate[j] == true_gold[j]:
            count += 1
    out = ''
    for j in range(d):
        out += ' {}'.format(idx2label[j])
    out += '\n'
    for j in range(d):
        out += idx2label[j]
        for k in range(d):
            out += ' {}'.format(str(int(m[j][k])))
        out += '\n'
    print("            {}\n Test accuracy={:.5f}\n".format(out, count/cand_len))

def evaluate_test_instances(data, model, sys_output, beam_size, topN, topK, features):
    instances, candidates, true_labels, sentence = {}, [], [], []
    b_index, index = 0, 0
    with open(sys_output, 'w') as w:
        w.write("\n\n%%%%% test data:\n")
        for word, gold, features in data:
            true_labels.append(gold)
            if index == model[3][b_index]: #new_sentence
                if sentence != []: 
                    cur_path, output = beam_decoder(sentence, instances) #generate op
                    candidates += cur_path
                    w.write(output)
                sentence = []
                index = 0
                b_index += 1
                instances = {}
                instances[index] = {}
                features["prevT=BOS"] = 1
                features["prevTwoTags=BOS+BOS"] = 1
                pos_topN = sorted(calculate_p(features,model).items(), key = lambda x:-x[1])[:topN]
                for tag in pos_topN:
                    instances[index][tag[0]] =  (tag[1], 'BOS', 1)
            else:
                instances[index] = {}
                for sequence in instances[index-1]:
                    if len(instances) == 1:
                        prevTag = 'BOS'
                    else:    
                        prevTag = instances[index-1][sequence][1]
                    features["prevT={}".format(sequence)] = 1
                    features["prevTwoTags={}+{}".format(sequence, prevTag)] = 1
                pos_topN = sorted(calculate_p(features,model).items(), key = lambda x:-x[1])[:topN] # take the topN probs 
                for tag in pos_topN:
                    #tuple of prob, prev_tag and prev_prob
                    instances[index][tag[0]] =  (tag[1], sequence, 1) 
                instances[index] = prune(beam_size, topK, instances[index])  
            sentence.append((word, gold))
            index += 1
        cur_path, output = beam_decoder(sentence, instances) 
        candidates += cur_path
        w.write(output)
    write_output(candidates, true_labels, num_features)

if __name__ == "__main__":
    s= time.time()
    if len(sys.argv) == 8:
        boundaries = read_boundary(sys.argv[2])
        data = read_data(sys.argv[1])
        model, num_features = read_model(sys.argv[3])
        model.append([0] + read_boundary(sys.argv[2]))
        evaluate_test_instances(data, model, sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), num_features)
        e=time.time()
        print("Running_time=:",(e-s)/60,min)