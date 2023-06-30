# ==============================================
# T02
#
# Sistemas Inteligentes - CSI30 - 2023/1
# Turma S71
# Jhonny Kristyan Vaz-Tostes de Assis - 2126672
# João Vítor Dotto Rissardi - 2126699
# ==============================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib 
import sys

test_file = sys.argv[1]

data = pd.read_csv(test_file, 
                  header=None, 
                  delimiter=',', 
                  usecols=[1, 2, 3], 
                  names=['qPA', 'pulso', 'resp'])

feature_cols = ['qPA', 'pulso', 'resp']

X = data.iloc[:]

sc_X  = StandardScaler()
X_scaled = sc_X.fit_transform(X)

model = joblib.load("nn_models/nn_best.pkl")

y_pred = model.predict(X_scaled) 

params = model.get_params()

file_name = "Neural Network Regressor"
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

print("============================================================================================================")
print(file_name, " | Activation: ", activation, 
    " \n| Solver: ", solver, " | Hidden Layer Sizes: ", hidden_layer_sizes,
    " \n| Alpha: ", alpha, " | Early Stopping: ", early_stopping,
    " \n| Learning Rate: ", learning_rate, " | Learning Rate Init: ", learning_rate_init,
    " \n| Max Iter: ", max_iter, " | Tol: ", tol,
    " \n| N Iter No Change: ", n_iter_no_change)
print("============================================================================================================")

print("RESULTS: ")

for value in y_pred:
      print(value)

with open('../Results/nn_results.txt', 'w') as file:
    for value in y_pred:
        file.write(str(value) + '\n')
