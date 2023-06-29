from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import joblib
import sys

data = pd.read_csv('../treino_sinais_vitais_com_label.txt', 
                   header=None, 
                   delimiter=',', 
                   usecols=[3, 4, 5, 6], 
                   names=['qPA', 'pulso', 'resp', 'gravidade'])

feature_cols = ['qPA', 'pulso', 'resp']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

sc_X  = StandardScaler()
X_trainscaled = sc_X.fit_transform(X_train)
X_testscaled = sc_X.fit_transform(X_test)

X_trainscaled = sc_X.fit_transform(X)
y_train = y

solver = ["sgd"]
activation = ["relu", "logistic", "tanh"]
momentum = [0.9]
tol = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
learning_rate = ['constant', 'adaptive', 'invscaling']
learning_rate_init = [1e-3, 5e-4, 1e-4]
alpha = [1e-4]
# random_state = [5, 10, 15, 20, 25]
max_iter = [300000]
n_iter_no_change = [20, 200, 2000]
verbose = [False]
early_stopping = [True, False]
hidden_layer_sizes = [(10, 10, 10), (50, 40, 20, 10), (10, 5, 5), (8, 9, 10), (20, 25, 15), (25, 30, 5), (15, 30), (15, 15, 15, 15)]

grid = dict(activation = activation, 
            tol = tol, 
            learning_rate = learning_rate, 
            learning_rate_init = learning_rate_init, 
            hidden_layer_sizes = hidden_layer_sizes,
            max_iter = max_iter,
            n_iter_no_change = n_iter_no_change,
            verbose = verbose,
            solver = solver)

model = MLPRegressor()

cvFold = RepeatedKFold(n_splits=10, n_repeats=4)
gridSearch = RandomizedSearchCV(estimator=model, param_distributions=grid, n_jobs=8, cv=cvFold, scoring="neg_mean_squared_error")

searchResults = gridSearch.fit(X_trainscaled, y)

bestModel = searchResults.best_estimator_

joblib.dump(bestModel, "bestmodel.pkl")

test_file = sys.argv[1]

data = pd.read_csv(test_file, 
                   header=None, 
                   delimiter=',', 
                   usecols=[3, 4, 5, 6], 
                   names=['qPA', 'pulso', 'resp', 'gravidade'])

feature_cols = ['qPA', 'pulso', 'resp']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_scaled = sc_X.fit_transform(X)

pred = bestModel.predict(X_scaled)

test_set_rmse = np.sqrt(mean_squared_error(y, pred))

print('RMSE: ', test_set_rmse)
print('Score: ', bestModel.score(X_scaled, y))

pred = bestModel.predict(X_trainscaled)

test_set_rmse = np.sqrt(mean_squared_error(y_train, pred))

print('RMSE: ', test_set_rmse)
print('Score: ', bestModel.score(X_scaled, y))


print("Params: ", bestModel.get_params())


# nn = MLPRegressor(
#     activation='relu',
#     solver='sgd',
#     hidden_layer_sizes=(500, 100, 10),
#     alpha=0.0001,
#     verbose=True,
#     early_stopping=True,
#     learning_rate='constant',
#     learning_rate_init=0.0005,
#     max_iter=30000,
#     tol=1e-7,
#     n_iter_no_change=10000
# )

# nn.fit(X_trainscaled, y_train)

# joblib.dump(nn, 'nn7.pkl')


# #nn = joblib.load('bestmodel.pkl')

# pred = nn.predict(X_testscaled)

# test_set_rsquared = r2_score(y_test, pred)
# test_set_rmse = np.sqrt(mean_squared_error(y_test, pred))

# print('R_squared value: ', test_set_rsquared)
# print('RMSE: ', test_set_rmse)

# pred = nn.predict(X_trainscaled)

# test_set_rsquared = r2_score(y_train, pred)
# test_set_rmse = np.sqrt(mean_squared_error(y_train, pred))

# print('R_squared value: ', test_set_rsquared)
# print('RMSE: ', test_set_rmse)

# test_file = sys.argv[1]

# data = pd.read_csv(test_file, 
#                    header=None, 
#                    delimiter=',', 
#                    usecols=[3, 4, 5, 6], 
#                    names=['qPA', 'pulso', 'resp', 'gravidade'])

# feature_cols = ['qPA', 'pulso', 'resp']

# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# X_scaled = sc_X.fit_transform(X)

# pred = nn.predict(X_scaled)

# test_set_rmse = np.sqrt(mean_squared_error(y, pred))
# print('RMSE: ', test_set_rmse)
# print(nn.score(X_scaled, y))