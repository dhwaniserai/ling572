import sys
import math

def read_model(model_file):
    model_txt = [] #list of lines in model file
    classes = []
    final_model = dict() #model dictionary
    cLabel = '' #current class label being read
    with open(model_file,'r') as f:
        model_str = f.read().strip()
        model_txt = model_str.split('\n')
    for line in model_txt:
        line = line.strip()
        if 'FEATURES FOR CLASS' in line:
            cIndex = len('FEATURES FOR CLASS')+1
            cLabel = line[cIndex:]
            classes.append(cLabel)
            final_model[cLabel] = dict()
        else:
            feature, val = line.split()[0], float(line.split()[1])
            final_model[cLabel][feature] = val
    return final_model, classes

def classify_inst(final_model, classes, features_di):
    results = dict()
    Z = 0
    for label in classes:
        val = final_model[label]["<default>"]
        for feat in features_di:
            if feat in final_model[label]:
                val += final_model[label][feat]
        results[label] = math.exp(val)
        Z += results[label]
    for label in results:
        results[label] /= Z

    return results

def read_instances(train_file):
    training_data = dict()
    lines = []
    with open(train_file, 'r') as f:
        lines = f.readlines()
    N = len(lines)
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        line_feats = line.split()
        cLabel = line_feats[0]
        features = set()
        for j in range(1, len(line_feats)):
            f, val = line_feats[j].split(":")
            features.add(f)
        if cLabel not in training_data:
            training_data[cLabel] = list()
        training_data[cLabel].append(features)
    return training_data, N

#calculating P(y|x)
def calc_cond_prob(training_data, model_file):
    data_counts = dict()
    feature_set = set()

    cond_prob_label = dict()
    for label in training_data: 
        cond_prob_label[label] = 1 / len(training_data)

    if model_file != None:
        final_model, classes = read_model(model_file)
    for label, lst_features in training_data.items():
        for feats in lst_features:
            if model_file != None:
                cond_prob_label = classify_inst(final_model, classes, feats)
            
            for feat in feats:
                feature_set.add(feat)
                for poss_label, cond_prob in cond_prob_label.items():
                    if poss_label not in data_counts:
                        data_counts[poss_label] = dict()
                    if feat not in data_counts[poss_label]:
                        data_counts[poss_label][feat] = 0
                    data_counts[poss_label][feat] += cond_prob

    for feature in feature_set:
        for label in data_counts:
            if feature not in data_counts[label]:
                data_counts[label][feature] = 0
    
    return data_counts

def output(output_file, data_counts, N):
    with open(output_file, 'w') as output:
        data_counts_lst = sorted(data_counts.items(), key = lambda key_value: key_value[0])
        for label_feat_count in data_counts_lst:
            clabel, feat_count = label_feat_count
            feat_count_lst = sorted(feat_count.items(), key = lambda key_value: key_value[0])
            for feat_count in feat_count_lst:
                feat, val = feat_count
                exp = val / N
                output.write(clabel + " " + feat + " " + str(round(exp, 5)) + " " + str(round(val, 5)) + "\n")

training_data = sys.argv[1]
output_file = sys.argv[2]

if len(sys.argv) > 3:
    model_file = sys.argv[3]
else:
    model_file = None


training_data, N = read_instances(training_data)
data_counts = calc_cond_prob(training_data, model_file)
output(output_file, data_counts, N)