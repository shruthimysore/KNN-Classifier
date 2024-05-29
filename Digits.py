import pandas as pd
from sklearn.model_selection import train_test_split
import Knn as knn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits(return_X_y=True)
digits_dataset_X = digits[0]
digits_dataset_y = digits[1]

# data = pd.read_csv("iris.csv", header=None)
training_accuracy = []
testing_accuracy = []
train_accuracy = []
test_accuracy = []
train_std = []
test_std = []
for k in range(1,52,2):
    for _ in range(1):
        X = digits_dataset_X
        Y = digits_dataset_y
        X = knn.normalize_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=0, shuffle=True)
        # print(X_train)
        # print(y_train)
        train_prediction = knn.predict(dataset1=X_train, dataset2=X_train, train_predictions=y_train, k=k)
        # print(train_prediction)
        # print(y_train)
        test_prediction = knn.predict(dataset1=X_train, dataset2=X_test, train_predictions=y_train, k=k)
        # print(test_prediction)
        # print(y_test)
        training_accuracy.append(knn.accuracy(true_data=y_train, predictions=train_prediction))
        testing_accuracy.append(knn.accuracy(true_data=y_test, predictions=test_prediction))
    train_accuracy.append(np.mean(training_accuracy))
    test_accuracy.append(np.mean(testing_accuracy))
    train_std.append(np.std(training_accuracy))
    test_std.append(np.std(testing_accuracy))
k = range(1,52,2)
plt.plot(k, train_accuracy)
plt.show()
plt.plot(k, test_accuracy)
plt.show()

