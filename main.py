import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

#adjust_labels_to_binary function get as srguments the training set target y_train as nparry,and a target calss value as string
#it reteruns an nparry with the same shape as y_train with only binary labels: 1 for the target class value and -1 otherwise
def adjust_labels_to_binary(y_train, target_class_value):
    pass

#one_vs_rest function gets as arguments x_train and y_train both as nparrays, and target_class_value as string
#it first binarize y_train according to target_class_value via the function adjust_labels_to_binary
#it returns a logistic regression model object trained on x_train and y_binarized     
def one_vs_rest(x_train, y_train, target_class_value):
    y_train_binarized = adjust_labels_to_binary(y_train, target_class_value)
    pass


#binarized_confusiom_matrix gets as arguments X, y_binarized as nparrays, the appropraite one_vs_rest_model as a model object
#and prob_threshold value
#it utilizes one_vs_rest model and predicted probabilities and the prob_threshold to predict
#y_pred on X
#it comparse it to the  y_binarized and 
#return an nparray of the appropriate confusion matrix as follows:
#[TP, FP
#FN, TN]    
def binarized_confusion_matrix(X, y_binarized, one_vs_rest_model, prob_threshold):
    pass


#micro_avg_precision gets as arguments X, y as nparrays, 
#all_target_class_dict a dictionary with key class value as string with value per key of the approprite one_vs_rest model
#prob_threshold the probability that if greater or equal to the prediction is 1, otherwise -1
#returns the micro average precision
def micro_avg_precision(X, y, all_target_class_dict, prob_threshold):
    pass



def micro_avg_recall(X, y, all_target_class_dict, prob_threshold):
    pass



def micro_avg_false_positve_rate(X, y, all_target_class_dict, prob_threshold):
    pass



def f_beta(precision, recall, beta):
    pass


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98, test_size=0.3)
