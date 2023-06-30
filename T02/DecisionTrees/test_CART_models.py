# ==============================================
# T02
#
# Sistemas Inteligentes - CSI30 - 2023/1
# Turma S71
# Jhonny Kristyan Vaz-Tostes de Assis - 2126672
# João Vítor Dotto Rissardi - 2126699
# ==============================================

import pandas as pd
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

model = joblib.load('CART_models/model_best.pkl')

y_pred = model.predict(X) 

params = model.get_params()

file_name = "CART Classifier"
model_max_depth = params["max_depth"]
model_min_samples = params["min_samples_leaf"]
criterion = params["criterion"]

print("============================================================================================================")
print("Modelo: ", file_name, " | Max Depth: ", model_max_depth, 
    " | Min Samples: ", model_min_samples, " | Criterion: ", criterion)
print("============================================================================================================")

print("RESULTS:")

for value in y_pred:
      print(value)

with open('../Results/CART_results.txt', 'w') as file:
    for value in y_pred:
        file.write(str(value) + '\n')
