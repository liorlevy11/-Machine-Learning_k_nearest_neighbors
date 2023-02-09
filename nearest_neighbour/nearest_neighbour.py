import random
import numpy as np
import pandas as pd
import matplotlib
from scipy.spatial import distance


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m alongside its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    m = len(y_train)
    classifier = [(x_train[i, :], y_train[i]) for i in range(m)]
    classifier.append(k)
    return classifier


def predictknn(classifier, x_test: np.array):
    """
    for each picture(cordinate) x_test:


    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k = classifier[-1]
    n = len(x_test)
    m = len(classifier) - 1
    answer = np.empty(n, dtype=float)
    for i in range(n):
        dist = [(distance.euclidean(classifier[j][0], x_test[i]), classifier[j][1]) for j in range(m)]
        dict = {}
        sortedByDistance = sorted(dist)
        firstKDistances = sortedByDistance[0:k]
        for dist_, label in firstKDistances:
            if dict.get(label) == None:
                dict[label] = 1
            else:
                dict.update({label: dict.get(label) + 1})
        predictedLabel = max(dict, key=dict.get)
        answer[i] = predictedLabel
    return answer.reshape(n, 1)


def returnAverageError(sampleSize: int,k=1,testSize=50, corrupted=False) -> float:
    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']
    averageOfAverages = 0.0
    for i in range(10):
        x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], sampleSize)
        if corrupted:
            y_train=corruptLabels(y_train)
        x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], testSize)

        classifer = learnknn(k, x_train, y_train)

        preds = predictknn(classifer, x_test)
        average = calculateAverage(preds, x_test, y_test)
        averageOfAverages += average
    return averageOfAverages / 10


def corruptLabels(y_train):
    length = len(y_train)
    all_indexes = [i for i in range(length)]
    random.shuffle(all_indexes)
    new_indexes = all_indexes[:int(length * 0.15)]
    for index in new_indexes:
        oldLabel = y_train[index]
        y_train[index] = chooseDifferentLabel(oldLabel)
    return y_train


def calculateAverage(preds, x_test, y_test):
    counter = 0
    for i in range(x_test.shape[0]):
        if y_test[i] != preds[i]:
            counter += 1
    average = (counter / x_test.shape[0])
    return average


def returnMaxError(sampleSize: int,k=1,testSize=50, corrupted=False) -> float:
    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']
    maxError = -999.0
    for i in range(10):
        x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], sampleSize)
        if corrupted:
            y_train = corruptLabels(y_train)
        x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], testSize)

        classifer = learnknn(k, x_train, y_train)

        preds = predictknn(classifer, x_test)
        average = calculateAverage(preds, x_test, y_test)
        if average > maxError:
            maxError = average
    return maxError


def returnMinError(sampleSize: int,k=1,testSize=50, corrupted=False) -> float:
    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']
    minError = 99999999999999999999.0
    for i in range(10):
        x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], sampleSize)
        if corrupted:
            y_train = corruptLabels(y_train)
        x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], testSize)

        classifer = learnknn(k, x_train, y_train)

        preds = predictknn(classifer, x_test)
        average = calculateAverage(preds, x_test, y_test)

        if average < minError:
            minError = average
    return minError

def chooseDifferentLabel(currentLabel: float)->float:
    while True:
        r = random.randint(0, 3)
        if r != currentLabel:
            return r

def Second_a():
    average = [returnAverageError(i) for i in range(20, 120, 20)]
    max_error = [returnMaxError(i) for i in range(20, 120, 20)]
    lowest_error = [returnMinError(i) for i in range(20, 120, 20)]
    index = ['20', '40', '60', '80', '100']
    yticks = [i for i in np.arange(0, 0.35, 0.025)]
    df = pd.DataFrame({'Average': average,
                       'Max Error': max_error,
                       'Lowest Error': lowest_error}, index=index)
    ax = df.plot.bar(rot=0)
    ax.set_yticks(yticks)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Error")
    ax.set_title("Error of the nn algorithm on Different sizes")
    # ax.tick_params(axis='y', labelsize=5)
    fig = ax.get_figure()
    fig.savefig("2a.png")

def Second_e(corrupted=False):
    average = [returnAverageError(200, i, 100,corrupted) for i in range(1,12)]
    max_error = [returnMaxError(200, i, 100,corrupted) for i in range(1,12)]
    lowest_error = [returnMinError(200, i, 100,corrupted) for i in range(1,12)]
    index = [f"{i}" for i in range(1,12)]
    yticks = [i for i in np.arange(0, 0.35, 0.025)]
    df = pd.DataFrame({'Average': average,
                       'Max Error': max_error,
                       'Lowest Error': lowest_error}, index=index)
    ax = df.plot.bar(rot=0)
    ax.set_yticks(yticks)
    ax.set_xlabel("k")
    ax.set_ylabel("Error")
    ax.set_title("Error of the knn algorithm on Different k")
    # ax.tick_params(axis='y', labelsize=5)
    fig = ax.get_figure()
    if(corrupted):
        fig.savefig("2f.png")
    else:
     fig.savefig("2e.png")

def Second_f():
    Second_e(True)

def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train2']
    train1 = data['train3']
    train2 = data['train5']
    train3 = data['train6']

    test0 = data['test2']
    test1 = data['test3']
    test2 = data['test5']
    test3 = data['test6']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"
    # print(np.mean(preds != y_test))
    # counter = 0
    # for i in range(x_test.shape[0]):
    #     if y_test[i] != preds[i]:
    #         counter += 1
    # print(counter/x_test.shape[0])
    #
    # k = 1
    # x_train = np.array([[1, 2], [3, 4], [5, 6]])
    # y_train = np.array([1, 0, 1])
    # classifier = learnknn(k, x_train, y_train)
    # x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    # y_testprediction = predictknn(classifier, x_test)
    # print(y_testprediction);
    # get a random example from the test set
    # i = np.random.randint(0, x_test.shape[0])
    # for i in range(len(preds)):
    # # this line should print the classification of the i'th test sample.
    #     print(f"The {i}'th test sample was classified as {preds[i]}")
    Second_a()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # Second_a()
    # Second_e()
    Second_f()

def assignment1c():
