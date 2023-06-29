import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import os
import joblib 
import sys
import numpy as np

#Load data

test_file = sys.argv[1]

data = pd.read_csv(test_file, 
                  header=None, 
                  delimiter=',', 
                  usecols=[3, 4, 5, 6], 
                  names=['qPA', 'pulso', 'resp', 'gravidade'])

feature_cols = ['qPA', 'pulso', 'resp']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

sc_X  = StandardScaler()
X_scaled = sc_X.fit_transform(X)


#Get all models

folder = "nn_models/"

prefix = "model"

files = os.listdir(folder)

#Print measuments for all models

for file in files:

      model = joblib.load(folder + file)

      y_pred = model.predict(X_scaled) 

      params = model.get_params()

      file_name = file.split(".")[0]
      activation = params["activation"]
      solver = params["solver"]
      hidden_layer_sizes = params["hidden_layer_sizes"]
      alpha = params["alpha"]
      early_stopping = params["early_stopping"]
      learning_rate = params["learning_rate"]
      learning_rate_init = params["learning_rate_init"]
      max_iter = params["max_iter"]
      tol = params["tol"]
      n_iter_no_change = params["n_iter_no_change"]
      model_index = file_name.split("_")[1]
      #model_ccp_alpha = int(file_name.split("_")[5])

      #Calculate metrics
      #accuracy = metrics.accuracy_score(y, y_pred)
      #precision = metrics.precision_score(y, y_pred, average='weighted', zero_division=0)
      #recall = metrics.recall_score(y,y_pred, average='weighted', zero_division=0)
      #weighted_average_f_measure = metrics.f1_score(y, y_pred, average='weighted')
      score = model.score(X_scaled, y)
      rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))

      # Print metrics
      print("============================================================================================================")
      print("Modelo: ", model_index, " | Activation: ", activation, 
            " \n| Solver: ", solver, " | Hidden Layer Sizes: ", hidden_layer_sizes,
            " \n| Alpha: ", alpha, " | Early Stopping: ", early_stopping,
            " \n| Learning Rate: ", learning_rate, " | Learning Rate Init: ", learning_rate_init,
            " \n| Max Iter: ", max_iter, " | Tol: ", tol,
            " \n| N Iter No Change: ", n_iter_no_change)
      print("============================================================================================================")
      print("Score:",score)
      print("RMSE: ", rmse)
      # for value in y_pred:
      #       print(value)
      #print("F-measure:", f_measure)
      #print("Average precision: ", average_precision)
      #print("Average recall: ", average_recall)
      #print("Weighted Average f-measure: ", weighted_average_f_measure)

#  Modelo:  2  | Activation:  relu
# | Solver:  sgd  | Hidden Layer Sizes:  (10, 10, 5, 4)
# | Momentum: 0.9
# | Alpha:  0.0001  | Early Stopping:  False
# | Learning Rate:  constant  | Learning Rate Init:  0.0005
# | Max Iter:  30000  | Tol:  1e-09
# | N Iter No Change:  150