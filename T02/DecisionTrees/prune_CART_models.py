# ==============================================
# T02
#
# Sistemas Inteligentes - CSI30 - 2023/1
# Turma S71
# Jhonny Kristyan Vaz-Tostes de Assis - 2126672
# João Vítor Dotto Rissardi - 2126699
# ==============================================

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.preprocessing import LabelBinarizer 
# from graphviz import Source
# from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import os
# import numpy as np
# from itertools import cycle
import joblib 
# from itertools import repeat, chain
# import random
# import warnings

#Load data

data = pd.read_csv('../treino_sinais_vitais_com_label.txt', 
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43) 

#Print measuments for all models

for file in files:

    model = joblib.load(folder + file)

    y_pred = model.predict(X_test) 

    file_name = file.split(".")[0]
    model_max_depth = int(file_name.split("_")[1])
    model_min_samples = int(file_name.split("_")[2])
    criterion = file_name.split("_")[3]
    model_index = int(file_name.split("_")[4])

    path = model.cost_complexity_pruning_path(X_train, y_train)

    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []

    for ccp_alpha in ccp_alphas:
        if criterion == "logloss":
            criterion = "log_loss"
        clf = DecisionTreeClassifier(max_depth=model_max_depth, min_samples_leaf=model_min_samples, criterion=criterion, ccp_alpha=ccp_alpha)
        clf = clf.fit(X_train, y_train)
        clfs.append(clf)
    
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    print("Ccp_alphas: ", ccp_alphas)

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy vs alpha for training and testing sets for model {model_index}")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    