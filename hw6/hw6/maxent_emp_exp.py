import sys
from collections import defaultdict

train_data = sys.argv[1]
output_file = sys.argv[2]

features = set() #set of features in the model
data_counts = dict() #dict of class feature counts

with open(train_data, 'r') as f:
    train_lines = f.readlines()
    N = len(train_lines)
    for line in train_lines:
        line = line.strip()
        if line == '':
            continue
        line_feats = line.split()
        cLabel = line_feats[0]
        for j in range(1, len(line_feats)):
            feature_count = line_feats[j].split(":")
            feature = feature_count[0]
            if cLabel not in data_counts:
                data_counts[cLabel] = defaultdict(int)
            data_counts[cLabel][feature] += 1
            features.add(feature)

for feature in features:
    for label in data_counts:
        if feature not in data_counts[label]:
            data_counts[label][feature] = 0

# write output #
with open(output_file, 'w') as output:
    data_counts_lst = sorted(data_counts.items(), key = lambda key_value: key_value[0])
    for label_feat_count in data_counts_lst:
        label, feat_count = label_feat_count
        feat_count_lst = sorted(feat_count.items(), key = lambda key_value: key_value[0])
        for feat_count in feat_count_lst:
            feature, count = feat_count
            exp = count / N
            output.write(label + " " + feature + " " + str(round(exp, 5)) + " " + str(count) + "\n")


        