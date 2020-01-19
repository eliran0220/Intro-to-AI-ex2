import sys


class NaiveBayes:
    def __init__(self, given_features, train, y_hat):
        """
        Initialization of the NaiveBayes model
        :param given_features: features
        :param train: The train set
        :param y_hat: The y hats
        """
        self.features = given_features.copy()
        del self.features[-1]
        self.train_set = train.copy()
        self.y_hat = y_hat
        self.total = len(self.y_hat)
        self.yes, self.no = self.count_yes_no()
        self.yes_probability = self.yes / (self.yes + self.no)
        self.no_probability = self.no / (self.yes + self.no)
        self.likelihood_table = self.create_likelihood_table()

    def create_frequency_table(self):
        """
        Creating the frequency table of each feature
        :return: the frequency table
        """
        table = {}
        feature_index = 0
        for feature in self.features:
            table[feature] = {}
            for example, tag in zip(self.train_set, self.y_hat):
                att = example[feature_index]
                if att in table[feature]:
                    table[feature][att][tag] += 1
                else:
                    table[feature][att] = {'yes': 0, 'no': 0}
                    table[feature][att][tag] += 1
            feature_index += 1
        return table

    def create_likelihood_table(self):
        """
        Creating the likelihood table, for each feature
        :return:
        """
        frequency_table = self.create_frequency_table()
        likelihood_table = {}
        for item in frequency_table:
            likelihood_table[item] = {}
            table = frequency_table[item]
            for att in table:
                p_att_yes = (table[att]['yes'] / self.total) / self.yes_probability
                p_att_no = (table[att]['no'] / self.total) / self.no_probability
                likelihood_table[item][att] = {'yes': p_att_yes, 'no': p_att_no}
        return likelihood_table

    def predict(self, to_predict):
        """
        The function predicts the tags
        :param to_predict: the test
        :return: the predictions
        """
        predictions = []
        if all(isinstance(i, list) for i in to_predict):
            for item in to_predict:
                predictions.append(self.find_tag(item))
            return predictions
        else:
            prediction = self.find_tag(to_predict)
            return prediction

    def find_tag(self, item):
        """
        The function finds the tag based on probability
        :param item: the item we want to tag
        :return: prediction (tag)
        """
        index = 0
        yes = 1
        no = 1
        for ex in item:
            current_feature = self.features[index]
            yes *= self.likelihood_table[current_feature][ex]['yes']
            no *= self.likelihood_table[current_feature][ex]['no']
            index += 1
        try:
            p_yes = (self.yes_probability * yes) / ((self.yes_probability * yes) + (self.no_probability * no))
            p_no = (self.no_probability * no) / ((self.yes_probability * yes) + (self.no_probability * no))
        except:
            sys.exit(-1)
        if p_yes > p_no:
            return 'yes'
        else:
            return 'no'

    def count_yes_no(self):
        """
        The function counts the number of yes and no
        :return: yes,no number
        """
        yes_counter = 0
        no_counter = 0
        for tag in self.y_hat:
            if tag == 'yes':
                yes_counter += 1
            else:
                no_counter += 1
        return yes_counter, no_counter

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
