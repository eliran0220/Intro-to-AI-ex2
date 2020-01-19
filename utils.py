import math
from KNN import KNN
from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes


def train_set_reader(data):
    """
    :param data: given training data
    :return: features, train_set, y_hat tags
    """
    features = None
    train_set = []
    y_hat = []
    with open(data) as file:
        features = file.readline().split('\t')
        features[-1] = features[-1].replace('\n', '')
        for line in file:
            example = line.split('\t')
            classification = example[-1].replace('\n', '')
            del example[-1]
            train_set.append(example)
            y_hat.append(classification)
    return features, train_set, y_hat


def test_set_reader(data):
    """
    :param data: given test data
    :return: train_set, y_hat tags
    """
    test_set = []
    y_hat = []
    with open(data) as file:
        features = file.readline().split('\t')  # no need features in test, so we read it
        for line in file:
            example = line.split('\t')
            classification = example[-1].replace('\n', '')
            del example[-1]
            test_set.append(example)
            y_hat.append(classification)
    return test_set, y_hat


def chunks(dataset, n):
    """
    The function creates chunks of the dataset, used for cross validation with 5 sets
    :param dataset: given dataset
    :param n: number of chunks to be made
    :return: chunks of dataset, splitted to n sets
    """
    length = len(dataset)
    # For item i in a range that is a length of l,
    for i in range(0, length, n):
        # Create an index range for l of n items:
        yield dataset[i:i + n]


def create_other_sets(i, zipped):
    """
    The function creates the other sets which are the training sets
    :param i: the iteration
    :param zipped: the zipped chunks
    :return:
    """
    other_set = []
    index = 0
    for list in zipped:
        if index == i:
            index += 1
        else:
            for item in list:
                other_set.append(item)
            index += 1
    return other_set


def k_fold_cross_validation(algo, features, train_set, y_hat):
    """

    :param algo: name of algorithm
    :param features: features
    :param train_set: the train_set
    :param y_hat: the tags
    :return: a list of all the acc, and the average acc
    """
    zipped = list(chunks(train_set, math.ceil(len(train_set) / 5)))
    y_hat_zipped = list(chunks(y_hat, math.ceil(len(y_hat) / 5)))
    accuracy_list = []
    average_accuracy = 0
    for i in range(5):
        acc = 0
        test_set = zipped[i]
        test_set_y_hat = y_hat_zipped[i]
        train_set_new = create_other_sets(i, zipped)
        train_set_y_hats_new = create_other_sets(i, y_hat_zipped)
        model = create_model(algo, features, train_set_new, train_set_y_hats_new)
        predictions = model.predict(test_set)
        length = len(predictions)
        for i in range(length):
            if predictions[i] == test_set_y_hat[i]:
                acc += 1
        acc = acc / length
        acc = truncate_number(acc, 2)
        average_accuracy += acc
        accuracy_list.append(acc)
    average_accuracy = average_accuracy / 5
    average_accuracy = truncate_number(average_accuracy, 2)
    return accuracy_list, average_accuracy


def truncate_number(n, decimals=0):
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


def create_accuracy_file(features, train, y_hat):
    """
    Create the accuracy.txt file
    :param features: features
    :param train: the train set
    :param y_hat: the y_hat set
    :return:
    """
    list_dt, avg_dt = k_fold_cross_validation('DecisionTree', features, train, y_hat)
    list_bayes, avg_bayes = k_fold_cross_validation('NaiveBayes', features, train, y_hat)
    list_knn, avg_knn = k_fold_cross_validation("KNN", features, train, y_hat)
    with open('accuracy.txt', 'w') as file:
        file.write(str(avg_dt) + '\t' + str(avg_knn) + '\t' + str(avg_bayes))


def create_model(algo, given_features, given_train_set, given_y_hat):
    """
    Given an algorithm, builds the model
    :param algo: algorithm
    :param given_features: features
    :param given_train_set: the train set
    :param given_y_hat: the y hat set
    :return:
    """
    model = None
    if algo == 'KNN':
        model = KNN(given_train_set, given_y_hat, 5)
    elif algo == 'NaiveBayes':
        model = NaiveBayes(given_features, given_train_set, given_y_hat)
    elif algo == 'DecisionTree':
        model = DecisionTree(given_features, given_train_set, given_y_hat)
    return model


def create_output(features, train_set, y_hat, test_set, y_hat_test_set):
    """
    Creates the output.txt file
    :param features: features
    :param train_set: the train set
    :param y_hat: the y hat set
    :param test_set: the test set
    :param y_hat_test_set: the y test hat set
    :return: nothing
    """
    list_avg = []
    knn = KNN(train_set, y_hat, 5)
    nb = NaiveBayes(features, train_set, y_hat)
    dt = DecisionTree(features, train_set, y_hat)
    avg_knn = knn.test(test_set, y_hat_test_set)
    avg_nb = nb.test(test_set, y_hat_test_set)
    avg_dt = dt.test(test_set, y_hat_test_set)
    list_avg.append(avg_dt)
    list_avg.append(avg_knn)
    list_avg.append(avg_nb)
    dt.print_tree('output.txt', list_avg)
