import random
from Node import Node
import math


class DecisionTree:
    def __init__(self, features, train_set, y_hat):
        """
        Initialize the decision tree model
        :param features: features
        :param train_set: the train set
        :param y_hat: the y hats
        """
        self.features = features.copy()
        del self.features[-1]
        self.feature_index = self.feature_index_dict()
        self.train_set = train_set
        self.y_hat = y_hat
        self.examples = self.build_examples_with_tags()
        self.feature_values_dict = self.init_feature_values()
        self.tree = self.build_tree()
        self.tree.set_is_root()
        self.tabs = 0

    def ID3(self, examples, attributes, default, value):
        """
        The ID3 algorithm, recursively builds the tree choosing the best att each time
        :param examples: the examples
        :param attributes: the attributes
        :param default: the default value
        :param value: the best value
        :return:
        """
        classification = self.check_classification(examples)
        if not examples:
            return Node(value, default, True)
        elif classification:
            return Node(value, classification, True)
        elif not attributes:
            new_mode = self.mode(examples)
            return Node(value, new_mode, True)
        else:
            best = self.choose_attribute(attributes, examples)
            tree = Node(best, is_leaf=False, index=self.feature_index[best])
            values = self.feature_values_dict[best]  # get all the values possible for this best attribute
            index = values[0]  # to get all the examples
            possible_values = self.get_values(values)
            for val in possible_values:
                new_examples = self.get_new_examples(index, val, examples)
                new_attributes = attributes.copy()  # get the new attributes without the current
                new_attributes.remove(best)
                new_default = self.mode(new_examples)
                sub_tree = self.ID3(new_examples, new_attributes, new_default, best)
                tree.add_child(val, sub_tree)
        return tree

    def feature_index_dict(self):
        """
        Builds a dictionary for each index of features
        :return: dictionary of indexes
        """
        dict = {}
        for i in range(len(self.features)):
            feature = self.features[i]
            dict[feature] = i
        return dict

    def get_values(self, values):
        """
        The function gets all the possible values from a feature
        :param values: list of values (with index at the start)
        :return: possible values of feature
        """
        val = []
        for i in range(1, len(values)):
            val.append(values[i])
        return val

    def get_new_examples(self, index, attribute, examples):
        """
        Builds the new examples from attribute and index
        :param index: the index
        :param attribute: the attribute
        :param examples: all the examples for now
        :return: new list of examples
        """
        new_examples = []
        for example in examples:
            if example[index] == attribute:
                new_examples.append(example)
        return new_examples

    def init_feature_values(self):
        """
        Initialization of the features to a dictionary, and
        for each feauture it's values
        :return: dictionary
        """
        feature_values_dict = {}
        length = len(self.features)
        for i in range(length):
            feature_values_dict[self.features[i]] = []
            feature_values_dict[self.features[i]].append(i)
            for example in self.train_set:
                att = example[i]
                if att not in feature_values_dict[self.features[i]]:
                    feature_values_dict[self.features[i]].append(att)
        feature_values_dict = self.sort_feature_dict(feature_values_dict)
        return feature_values_dict

    def sort_feature_dict(self, dictionary):
        """
        A helper function to sort the dictionary with values by alphabet
        :param dictionary: a given dict
        :return: sorted dict
        """
        for value in dictionary:
            list_attributes = dictionary[value]
            index = list_attributes[0]
            list_attributes.remove(index)
            list_attributes.sort()
            list_attributes.insert(0, index)
            dictionary[value] = list_attributes
        return dictionary

    def mode(self, examples):
        """
        :param examples: all examples
        :return: if yes equals no, uniformly choose, else,
        the max between them
        """
        yes, no = self.get_yes_no_labels_count(examples)
        if yes == no:
            vec = ['yes', 'no']
            decide = random.choice(vec)
            if decide == 'yes':
                return 'yes'
            else:
                return 'no'
        if yes > no:
            return 'yes'
        else:
            return 'no'

    def get_yes_no_labels_count(self, examples):
        """
        Counting number of yes and no
        :param examples: all examples
        :return: yes,no numbers
        """
        num_yes = 0
        num_no = 0
        for example in examples:
            if example[-1] == 'yes':
                num_yes += 1
            else:
                num_no += 1
        return num_yes, num_no

    def check_classification(self, examples):
        """
        Checks if all tags are yes or no
        :param examples: all examples
        :return: 'yes', or 'no' if all examples are tagge like that,
        else False
        """
        if all([example[-1] == 'yes' for example in examples]):
            return 'yes'
        elif all([example[-1] == 'no' for example in examples]):
            return 'no'
        else:
            return False

    def build_tree(self):
        """
        Builds the rec tree
        :return: the tree
        """
        default = self.mode(self.examples)
        self.tree = self.ID3(self.examples, self.features, default, None)
        return self.tree

    def choose_attribute(self, attributes, examples):
        """
        Choosing the best attributes given the examples,
        by calculating the total entropy, and the entropy for each feature
        :param attributes: attributes
        :param examples: examples
        :return:
        """
        total_yes, total_no = self.get_yes_no_labels_count(examples)
        total_entropy = self.calculate_entropy(total_yes, total_no)
        ig_dictionary = {}
        for feature in attributes:
            dictionary = {}
            weighted_avg = 0
            index = self.feature_values_dict[feature][0]
            for example in examples:
                att = example[index]  # we calculate how much for each attribute yes/no
                if att in dictionary:
                    dictionary[att][example[-1]] += 1
                else:
                    dictionary[att] = {'yes': 0, 'no': 0}
                    dictionary[att][example[-1]] += 1
            for att in dictionary:
                yes = dictionary[att]['yes']
                no = dictionary[att]['no']
                entropy = self.calculate_entropy(yes, no)
                weighted_avg += entropy * ((yes + no) / (total_yes + total_no))
            information_gain = total_entropy - weighted_avg
            ig_dictionary[feature] = information_gain
        return max(ig_dictionary, key=ig_dictionary.get)

    def build_examples_with_tags(self):
        """
        Builds examples with tags
        :return: examples with tags
        """
        examples_tags = []
        index = 0
        for example, tag in zip(self.train_set, self.y_hat):
            copied_example = example.copy()
            examples_tags.append(copied_example)
            examples_tags[index].insert(len(examples_tags[0]), tag)
            index += 1
        return examples_tags

    def calculate_entropy(self, yes, no):
        """
        Calculating the entropy
        :param yes: number yes
        :param no: number no
        :return: entropy
        """
        if yes == 0 or no == 0:
            return 0
        else:
            total = yes + no
            entropy = -(yes / total) * math.log(yes / total, 2) - (no / total) * math.log(no / total, 2)
            return entropy

    def print_tree(self, name, avg_list):
        """
        Prints the tree, or the output
        :param name: file name
        :param avg_list: if exists, the avg list of all models
        :return:
        """
        if name == 'tree.txt':
            file = open(name, 'w')
            self.print_iterative(file, self.tree, self.tabs)
            file.close()
        else:
            file = open(name, 'w')
            self.print_iterative(file, self.tree, self.tabs)
            file.write(str(avg_list[0]))
            file.write('\t' + str(avg_list[1]))
            file.write('\t' + str(avg_list[2]))
            file.close()

    def print_iterative(self, file, root, tabs):
        """
        Helper function to print iterativiely the tree
        :param file: the file to print to
        :param root: the root of tree
        :param tabs: tabs number
        :return: nothing
        """
        childrens = root.get_childrens()
        for child in childrens:
            if childrens[child].is_leaf():
                file.write(self.print_tabs(tabs))
                file.write("|" + root.get_value() + "=" + child + ":" + childrens[child].get_default() + '\n')
            else:
                file.write(self.print_tabs(tabs))
                value = root.get_value()
                if not root.is_root:
                    file.write("|" + value + "=" + child + '\n')
                    tabs += 1
                    self.print_iterative(file, childrens[child], tabs)
                    tabs -= 1
                else:
                    file.write(value + "=" + child + '\n')
                    tabs += 1
                    self.print_iterative(file, childrens[child], tabs)
                    tabs -= 1

    def print_tabs(self, tabs):
        """
        Makes a string of tabs
        :param tabs: number tabs
        :return: number
        """
        tabs_string = ''
        for i in range(tabs):
            tabs_string += '\t'
        return tabs_string

    def predict(self, to_predict):
        """
        The function predicts the tags
        :param to_predict: the test
        :return: the predictions
        """
        predictions = []
        if all(isinstance(i, list) for i in to_predict):
            for item in to_predict:
                predictions.append(self.find_tag(self.tree, item, 'default'))
            return predictions
        else:
            prediction = self.find_tag(self.tree, to_predict, 'default')
            return prediction

    def find_tag(self, tree, item, tag):
        """
        The function finds the tag based on the item and the tag
        :param item: the item we want to tag
        :return: prediction (tag)
        """
        if tag == 'yes':
            return 'yes'
        elif tag == 'no':
            return 'no'
        else:
            value = tree.get_value()
            index = tree.get_index()
            item_value = item[index]
            childrens = tree.get_childrens()
            next_node = childrens[item_value]
            tag = next_node.get_default()
            return self.find_tag(next_node, item, tag)

    def test(self, test_set, y_hat_test):
        """
        Given a test set, and the tags the function makes the accuracy of predictions
        :param test_set: the test set
        :param y_hat_test: the y hats
        :return: predictions accuracy
        """
        predictions = self.predict(test_set)
        index = 0
        acc = 0
        for tag in predictions:
            if tag == y_hat_test[index]:
                acc += 1
        acc = acc / len(y_hat_test)
        acc = self.truncate_number(acc, 2)
        return acc

    def truncate_number(self, n, decimals):
        """
        Round the number 2 points after decimel
        :param n: the number
        :param decimals: how much to cut after decimel
        :return: the number truncuated
        """
        multiplier = 10 ** decimals
        number = int(n * multiplier) / multiplier
        k = str(number)
        if len(k) == 3:
            number += 0.01
        return number
