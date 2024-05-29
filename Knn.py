import math as math
import numpy as np


def predict(dataset1, dataset2, train_predictions, k):
    predictions = []
    for instance in dataset2:
        predictions.append(find_neighbors_using_knn(training_set=dataset1, test_instance=instance, train_predictions=train_predictions, k=k))
    return predictions


def find_neighbors_using_knn(training_set, test_instance, train_predictions, k):
    distances = []
    for i in range(len(training_set)):
        distance = euclidean_distance(instance1=training_set[i], instance2=test_instance)
        distances.append((distance, train_predictions[i]))
    distances.sort(key=lambda x:x[0])
    # print(distances)
    neighbors = distances[:k]
    predicted_labels = {}
    for _, label in neighbors:
        if label in predicted_labels:
            predicted_labels[label] += 1
        else:
            predicted_labels[label] = 1
    majority_label = max(predicted_labels, key=predicted_labels.get)
    return majority_label


def euclidean_distance(instance1, instance2):
    # print(instance1)
    # print(instance2)
    squared_distance = 0
    for i in range(len(instance1)):
        squared_distance += (instance1[i] - instance2[i]) ** 2
    distance = math.sqrt(squared_distance)
    return distance


def accuracy(true_data, predictions):
    total = 0
    for i in range(len(predictions)):
        if true_data[i] == predictions[i]:
            total += 1
    accuracy = total/len(predictions)
    return accuracy


def normalize_data(dataset):
    X_min = np.min(dataset)
    X_max = np.max(dataset)
    X_normalized = (dataset - X_min)/(X_max - X_min)
    return X_normalized



