import sys
import math

def read_instances(data):
    training_data_dict = dict()
    with open(data, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            line_array = line.split()
            label = line_array[0]
            features = set()
            for j in range(1, len(line_array)):
                feature_count = line_array[j].split(":")
                feature = feature_count[0]
                count  = int(feature_count[1])
                features.add(feature)
            if label not in training_data_dict:
                training_data_dict[label] = list()
            training_data_dict[label].append(features)
    return training_data_dict, N

def read_in_model(model_file):
    model_dict = dict()

    with open(model_file, 'r') as f:
        lines = f.readlines()

    label = None
    for line in lines:
        line = line.strip()

        if "FEATURES FOR CLASS" in line:
            label = line[19:]
            model_dict[label] = dict()
        else:
            feature_prob = line.split()
            feature = feature_prob[0]
            prob = float(feature_prob[1])
            model_dict[label][feature] = prob

    return model_dict

def get_conditional_prob(training_data_dict, model_file):
    raw_count = dict()
    feature_set = set()

    cond_prob_per_label = dict()
    for label in training_data_dict: 
        cond_prob_per_label[label] = 1 / len(training_data_dict)

    if model_file != None:
        model_dict = read_in_model(model_file)
    for label, lst_features in training_data_dict.items():
        for features in lst_features:
            if model_file != None:
                cond_prob_per_label = find_from_model_file(model_dict, features)
            
            for feature in features:
                feature_set.add(feature)
                for possible_label, cond_prob in cond_prob_per_label.items():
                    if possible_label not in raw_count:
                        raw_count[possible_label] = dict()
                    if feature not in raw_count[possible_label]:
                        raw_count[possible_label][feature] = 0
                    raw_count[possible_label][feature] += cond_prob

    for feature in feature_set:
        for label in raw_count:
            if feature not in raw_count[label]:
                raw_count[label][feature] = 0
    
    return raw_count
        

def find_from_model_file(model_dict, features):
    result = dict()
    Z = 0
    for label in model_dict.keys():
        total = model_dict[label]["<default>"]
        for feature in features:
            if feature in model_dict[label]:
                total += model_dict[label][feature]
        result[label] = math.exp(total)
        Z += result[label]
    
    for label in result:
        result[label] /= Z 
    return result

def output(output_file, raw_count, N):
    with open(output_file, 'w') as output:
        raw_count_lst = sorted(raw_count.items(), key = lambda key_value: key_value[0])
        for label_feature_count in raw_count_lst:
            label, feature_count = label_feature_count
            feature_count_lst = sorted(feature_count.items(), key = lambda key_value: key_value[0])
            for feature_count in feature_count_lst:
                feature, count = feature_count
                exp = count / N
                output.write(label + " " + feature + " " + str(round(exp, 5)) + " " + str(round(count, 5)) + "\n")

training_data = sys.argv[1]
output_file = sys.argv[2]

if len(sys.argv) > 3:
    model_file = sys.argv[3]
else:
    model_file = None


training_data_dict, N = read_instances(training_data)
raw_count = get_conditional_prob(training_data_dict, model_file)
output(output_file, raw_count, N)


