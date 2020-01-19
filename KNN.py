class KNN:
    def __init__(self, train_set, y_hat, k):
        """
        The initialization of knn
        :param train_set: the train set
        :param y_hat: the y hat set
        :param k: the k neighbors
        """
        self.train_set = train_set.copy()
        self.y_hat = y_hat
        self.k = k

    def hamming_distance(self, str1, str2):
        """
        The function computes and returns the hamming distance between
        two strings
        :param str1: first string
        :param str2: second string
        :return: hamming distance
        """
        return sum(1 for (a, b) in zip(str1, str2) if a != b)

    def predict(self,to_predict):
        predictions = []
        if all(isinstance(i, list) for i in to_predict):
            for item in to_predict:
                predictions.append(self.find_tag(item))
            return predictions
        else:
            prediction = self.find_tag(to_predict)
            return prediction

    def find_tag(self, to_predict):
        """
        The function preficts the tag given an item in test set
        :param test_set: given test set
        :return: predictions
        """
        predictions = []
        distances = []
        neighbors = []
        yes_no_dict = {"yes": 0, "no": 0}
        y_hat_index = 0
        for example in self.train_set:
            distance = self.hamming_distance(to_predict, example)
            distances.append((example, self.y_hat[y_hat_index], distance))
            y_hat_index += 1
        distances.sort(key=lambda value: value[2])
        for i in range(self.k):
            neighbors.append(distances[i])
        for neighbor in neighbors:
            yes_no_dict[neighbor[1]] += 1
        max_value = max(yes_no_dict, key=yes_no_dict.get)
        return max_value

    def test(self, test_set, y_hat_test_set):
        """
        Given a test set and it's tags, computes the acc
        :param test_set:
        :param y_hat_test_set:
        :return:
        """
        predictions = self.predict(test_set)
        index = 0
        acc = 0
        for tag in predictions:
            if tag == y_hat_test_set[index]:
                acc += 1
        acc = acc / len(y_hat_test_set)
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
