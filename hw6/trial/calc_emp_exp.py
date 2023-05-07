import sys

training_data = sys.argv[1]
output_file = sys.argv[2]

feature_set = set()

raw_count = dict()

with open(training_data, 'r') as f:
    lines = f.readlines()
    N = len(lines)
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        line_array = line.split()
        true_class_label = line_array[0]
        for j in range(1, len(line_array)):
            feature_count = line_array[j].split(":")
            feature = feature_count[0]
            if true_class_label not in raw_count:
                raw_count[true_class_label] = dict()
            if feature not in raw_count[true_class_label]:
                raw_count[true_class_label][feature] = 0
            raw_count[true_class_label][feature] += 1
            feature_set.add(feature)

for feature in feature_set:
    for label in raw_count:
        if feature not in raw_count[label]:
            raw_count[label][feature] = 0

## output_file ##

with open(output_file, 'w') as output:
    raw_count_lst = sorted(raw_count.items(), key = lambda key_value: key_value[0])
    for label_feature_count in raw_count_lst:
        label, feature_count = label_feature_count
        feature_count_lst = sorted(feature_count.items(), key = lambda key_value: key_value[0])
        for feature_count in feature_count_lst:
            feature, count = feature_count
            exp = count / N
            output.write(label + " " + feature + " " + str(round(exp, 5)) + " " + str(count) + "\n")


        