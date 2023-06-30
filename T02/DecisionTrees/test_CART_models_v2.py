# ==============================================
# T02
#
# Sistemas Inteligentes - CSI30 - 2023/1
# Turma S71
# Jhonny Kristyan Vaz-Tostes de Assis - 2126672
# João Vítor Dotto Rissardi - 2126699
# ==============================================

import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
from sklearn import metrics
# from sklearn.preprocessing import LabelBinarizer 
# from graphviz import Source
# from sklearn.tree import export_graphviz
# import matplotlib.pyplot as plt
import os
import joblib 
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

folder = "CART_models/"

prefix = "model"

files = os.listdir(folder)

#Print measuments for all models

for file in files:

      model = joblib.load(folder + file)

      y_pred = model.predict(X) 

      params = model.get_params()

      file_name = file.split(".")[0]
      model_max_depth = params["max_depth"]
      model_min_samples = params["min_samples_leaf"]
      criterion = params["criterion"]
      model_index = file_name.split("_")[1]
      #model_ccp_alpha = int(file_name.split("_")[5])

      #Calculate metrics
      accuracy = metrics.accuracy_score(y, y_pred)
      precision = metrics.precision_score(y, y_pred, average='weighted', zero_division=0)
      recall = metrics.recall_score(y,y_pred, average='weighted', zero_division=0)
      weighted_average_f_measure = metrics.f1_score(y, y_pred, average='weighted')

      # Print metrics
      print("============================================================================================================")
      print("Modelo: ", model_index, " | Max Depth: ", model_max_depth, 
            " | Min Samples: ", model_min_samples, " | Criterion: ", criterion)
      print("============================================================================================================")
      print("Accuracy:",accuracy)
      print("Weighted Average Precision:",precision)
      print("Weighted Average Recall:",recall)
      print("Confusion Matrix:\n",metrics.confusion_matrix(y,y_pred))
      print("Weighted Average f-measure: ", weighted_average_f_measure)

      # for value in y_pred:
      #       print(value)

#Modelo:  2034  | Max Depth:  9  | Min Samples:  2  | Criterion:  log_loss