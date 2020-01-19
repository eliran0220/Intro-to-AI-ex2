from utils import train_set_reader, test_set_reader, create_output
from DecisionTree import DecisionTree
from KNN import KNN
import sys


def main(argv):
    train = argv[1]
    test = argv[2]
    features, train_set, y_hat = train_set_reader(train)
    test_set, y_hat_test = test_set_reader(test)
    create_output(features, train_set, y_hat, test_set, y_hat_test)



if __name__ == '__main__':
    main(sys.argv)
