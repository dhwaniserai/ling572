import sys
import math
from operator import itemgetter

class MaxEnt:

    def __init__(self, test_data, model_file, sys_output):
        self.model = self.read_in_model(model_file)
        self.classify(sys_output, test_data)

    def read_in_model(self, model_file):
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

    def classify(self, sys_output, test_data):
        labels = list(self.model.keys())

        sys_str = "%%%%% test data:\n"
        string, test_confusion, test_accuracy = self.read_instances(test_data, labels)
        sys_str += string

        with open(sys_output, 'w') as f:
            f.write(sys_str)

        self.evaluation(labels, test_confusion, test_accuracy)

    def read_instances(self, data, labels):
        # [ [class class class] [] [] ] same features in order of list
        confusion = [[0]*len(labels) for i in range(len(labels))]
        accuracy = 0

        sys_str = ""
        with open(data, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                line_array = line.split()
                label = line_array[0]
                features = dict()
                for j in range(1, len(line_array)):
                    feature_count = line_array[j].split(":")
                    feature = feature_count[0]
                    count  = int(feature_count[1])
                    features[feature] = count
                sys_str += "array:" + str(i) + " "
                string, predicted_label = self.classify_instance(features)
                sys_str += string
                confusion[labels.index(label)][labels.index(predicted_label)] += 1
                if label == predicted_label:
                    accuracy += 1
                i += 1
        accuracy /= len(lines)
        return sys_str, confusion, accuracy

    def classify_instance(self, features):
        result = dict()
        Z = 0
        for label in self.model.keys():
            total = self.model[label]["<default>"]
            for feature in features:
                if feature in self.model[label]:
                    total += self.model[label][feature]
            result[label] = math.exp(total)
            Z += result[label]
        
        for label in result:
            result[label] /= Z

        prob_classifications = list(result.items())

        prob_classifications = sorted(prob_classifications, key=itemgetter(1), reverse=True)
        sys_str = prob_classifications[0][0] + " "
        for classify in prob_classifications:
            sys_str += classify[0] + " " + str(round(classify[1], 5)) + " "
        sys_str += "\n"
        return sys_str, prob_classifications[0][0]

    def evaluation(self, labels, test_confusion, test_accuracy):
        eval_str = "Confusion matrix for the test data:\nrow is the truth, column is the system output\n\n             "
        for label in labels:
            eval_str += label + " "
        eval_str += "\n"
        for i in range(len(labels)):
            eval_str += labels[i] + " "
            for num in  test_confusion[i]:
                eval_str += str(num) + " "
            eval_str += "\n"
        eval_str += "\n"
        eval_str += " Test accuracy=" + str(round(test_accuracy, 5)) + "\n\n\n"
        print(eval_str)



if __name__ == "__main__":
    test_data = sys.argv[1]
    model_file = sys.argv[2]
    sys_output = sys.argv[3]

model = MaxEnt(test_data, model_file, sys_output)