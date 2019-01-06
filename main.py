import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris_dict = {}

#adjust_labels_to_binary function get as srguments the training set target y_train as nparry,and a target calss value as string
#it reteruns an nparry with the same shape as y_train with only binary labels: 1 for the target class value and -1 otherwise
def adjust_labels_to_binary(y_train, target_class_value):
    label_int = iris_dict[target_class_value]
    res = np.zeros(len(y_train))
    for i in range(len(y_train)):
        if y_train[i] == label_int:
            res[i] = 1
        else:
            res[i] = -1
    return res

#one_vs_rest function gets as arguments x_train and y_train both as nparrays, and target_class_value as string
#it first binarize y_train according to target_class_value via the function adjust_labels_to_binary
#it returns a logistic regression model object trained on x_train and y_binarized     
def one_vs_rest(x_train, y_train, target_class_value):
    y_train_binarized = adjust_labels_to_binary(y_train, target_class_value)
    clf = LogisticRegression(class_weight='balanced', solver='lbfgs')
    clf.fit(x_train, y_train_binarized)
    return clf


#binarized_confusiom_matrix gets as arguments X, y_binarized as nparrays, the appropraite one_vs_rest_model as a model object
#and prob_threshold value
#it utilizes one_vs_rest model and predicted probabilities and the prob_threshold to predict
#y_pred on X
#it comparse it to the  y_binarized and 
#return an nparray of the appropriate confusion matrix as follows:
#[TP, FN
#FP, TN]
def binarized_confusion_matrix(X, y_binarized, one_vs_rest_model, prob_threshold):
    y_prob = one_vs_rest_model.predict_proba(X)
    y_pred = [1 if y_prob[i][1] > prob_threshold else -1 for i in range(len(y_prob))]
    TP = sum([1 if y_pred[i] == y_binarized[i] and y_pred[i] == 1 else 0 for i in range(len(y_pred))])
    TN = sum([1 if y_pred[i] == y_binarized[i] and y_pred[i] == -1 else 0 for i in range(len(y_pred))])
    FP = sum([1 if y_pred[i] != y_binarized[i] and y_pred[i] == 1 else 0 for i in range(len(y_pred))])
    FN = sum([1 if y_pred[i] != y_binarized[i] and y_pred[i] == -1 else 0 for i in range(len(y_pred))])
    return np.array([[TP, FN], [FP, TN]])


#micro_avg_precision gets as arguments X, y as nparrays, 
#all_target_class_dict a dictionary with key class value as string with value per key of the approprite one_vs_rest model
#prob_threshold the probability that if greater or equal to the prediction is 1, otherwise -1
#returns the micro average precision
def micro_avg_precision(X, y, all_target_class_dict, prob_threshold):
    sum_TP = 0
    sum_FP = 0
    for flower in all_target_class_dict.keys():
        y_bin = adjust_labels_to_binary(y, flower)
        cnf_mat = binarized_confusion_matrix(X, y_bin, all_target_class_dict[flower], prob_threshold)
        sum_TP += cnf_mat[0][0]
        sum_FP += cnf_mat[1][0]
    return sum_TP / (sum_TP + sum_FP)


def micro_avg_recall(X, y, all_target_class_dict, prob_threshold):
    sum_TP = 0
    sum_FN = 0
    for flower in all_target_class_dict.keys():
        y_bin = adjust_labels_to_binary(y, flower)
        cnf_mat = binarized_confusion_matrix(X, y_bin, all_target_class_dict[flower], prob_threshold)
        sum_TP += cnf_mat[0][0]
        sum_FN += cnf_mat[0][1]
    return sum_TP / (sum_TP + sum_FN)




def micro_avg_false_positve_rate(X, y, all_target_class_dict, prob_threshold):
    sum_TN = 0
    sum_FP = 0
    for flower in all_target_class_dict.keys():
        y_bin = adjust_labels_to_binary(y, flower)
        cnf_mat = binarized_confusion_matrix(X, y_bin, all_target_class_dict[flower], prob_threshold)
        sum_TN += cnf_mat[1][1]
        sum_FP += cnf_mat[1][0]
    return sum_FP / (sum_TN + sum_FP)


def f_beta(precision, recall, beta):
    return (1+beta**2)*((precision*recall)/((beta**2*precision)+recall))


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98, test_size=0.3)
    for i in range(len(iris.target_names)):
        iris_dict[iris.target_names[i]] = i
    for flower in iris_dict.keys():
        clf = one_vs_rest(X_train, y_train, flower)
        print("Confusion matrix for {} train data".format(flower))
        y_train_bin = adjust_labels_to_binary(y_train, flower)
        mat = binarized_confusion_matrix(X_train, y_train_bin, clf, 0.5)
        print(mat)
        y_test_bin = adjust_labels_to_binary(y_test, flower)
        mat2 = binarized_confusion_matrix(X_test, y_test_bin, clf, 0.5)
        print("Confusion matrix for {} test data".format(flower))
        print(mat2)
    clf_dict = {}
    for flower in iris_dict.keys():
        clf_dict[flower] = one_vs_rest(X_train, y_train, flower)
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    x_axis = [micro_avg_false_positve_rate(X_train, y_train, clf_dict, prob) for prob in thresholds]
    y_axis = [micro_avg_recall(X_train, y_train, clf_dict, prob) for prob in thresholds]
    plt.figure(1)
    plt.title("ROC Curve")
    plt.xlabel('u average FP rate')
    plt.ylabel('u average recall')
    plt.plot(x_axis, y_axis)
    plt.show()
    T03_recall = micro_avg_recall(X_train, y_train, clf_dict, 0.3)
    T05_recall = micro_avg_recall(X_train, y_train, clf_dict, 0.5)
    T07_recall = micro_avg_recall(X_train, y_train, clf_dict, 0.7)
    T03_precision = micro_avg_precision(X_train, y_train, clf_dict, 0.3)
    T05_precision = micro_avg_precision(X_train, y_train, clf_dict, 0.5)
    T07_precision = micro_avg_precision(X_train, y_train, clf_dict, 0.7)
    betas = list(range(11))
    fb03 = [f_beta(T03_precision, T03_recall, b) for b in betas]
    fb05 = [f_beta(T05_precision, T05_recall, b) for b in betas]
    fb07 = [f_beta(T07_precision, T07_recall, b) for b in betas]
    plt.figure(2)
    plt.title('f_beta')
    plt.xlabel('beta')
    plt.ylabel('f_beta')
    plt.plot(betas, fb03, label="Threshold 0.3", color='red')
    plt.plot(betas, fb05, label="Threshold 0.5", color='green')
    plt.plot(betas, fb07, label="Threshold 0.7", color='blue')
    plt.legend(loc='best')
    plt.show()


