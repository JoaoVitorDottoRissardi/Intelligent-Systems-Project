#PARA REDES NEURAIS USAR SCIKIT LEARN COM SOLVER: SGD = STOCHASTIC GRAD DESCENT ;  ALGO BACKP

import pandas as pd
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer 
from graphviz import Source
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import os
import numpy as np
from itertools import cycle
import joblib 
from itertools import repeat, chain
import random
import warnings
import sys

#Load data

test_file = sys.argv[1]

data = pd.read_csv(test_file, 
                   header=None, 
                   delimiter=',', 
                   usecols=[3, 4, 5, 7], 
                   names=['qPA', 'pulso', 'resp', 'classe'])

feature_cols = ['qPA', 'pulso', 'resp']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#Get all models

folder = "id3_models/"

prefix = "model"

files = os.listdir(folder)

#Print measuments for all models

for file in files:

    model = joblib.load(folder + file)

    y_pred = model.predict(X) 

    file_name = file.split(".")[0]
    model_max_depth = file_name.split("_")[1]
    model_min_samples = file_name.split("_")[2]
    criterion = file_name.split("_")[3]
    model_index = int(file_name.split("_")[4])
    #model_ccp_alpha = int(file_name.split("_")[5])

     #Calculate metrics
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average=None, zero_division=0)
    recall = metrics.recall_score(y,y_pred, average=None, zero_division=0)
    f_measure = [(2 * p * r)/(p + r) if (p!=0 and r!=0) else 0 for p, r in zip(precision, recall)]
    average_precision = sum(precision) / float(len(precision))
    average_recall = sum(recall) / float(len(recall))
    average_f_measure = sum(f_measure) / float(len(f_measure))

    # Print metrics
    print("============================================================================================================")
    print("Modelo: ", model_index, " | Max Depth: ", model_max_depth, 
          " | Min Samples: ", model_min_samples, " | Criterion: ", criterion, " | ccp_alpha: ")#(model_ccp_alpha)
    print("============================================================================================================")
    print("Accuracy:",accuracy)
    print("Precision:",precision)
    print("Recall:",recall)
    print("Confusion Matrix:\n",metrics.confusion_matrix(y,y_pred))
    print("F-measure:", f_measure)
    print("Average precision: ", average_precision)
    print("Average recall: ", average_recall)
    print("Average f-measure: ", average_f_measure)