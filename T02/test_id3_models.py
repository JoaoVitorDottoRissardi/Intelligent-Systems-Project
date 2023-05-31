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

#Load data

data = pd.read_csv('treino_sinais_vitais_com_label.txt', 
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

i = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43) 

#Print measuments for all models

for file in files:

    model = joblib.load(folder + file)

    y_pred = model.predict(X_test) 

    file_name = file.split(".")[0]
    model_max_depth = file_name.split("_")[1]
    model_min_samples = file_name.split("_")[2]
    criterion = file_name.split("_")[3]

    # Model Accuracy, how often is the classifier correct?
    print("==========================================================================")
    print("Modelo: ", i, " | Max Depth: ", model_max_depth, " | Min Samples: ", model_min_samples, " | Criterion: ", criterion)
    print("==========================================================================")
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred, average=None, zero_division=0))
    print("Recall Score:",metrics.recall_score(y_test,y_pred, average=None, zero_division=0))
    print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_pred))
    
    i += 1