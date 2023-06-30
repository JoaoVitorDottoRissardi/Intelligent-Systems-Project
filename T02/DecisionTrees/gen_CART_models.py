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
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn import metrics
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import sys
import joblib 
import csv

# Get parameters from user
# max_depth = int(sys.argv[1])
# min_samples = int(sys.argv[2])
# criterion = sys.argv[3]

option = sys.argv[1]

#Load data
data = pd.read_csv('../treino.txt', 
                   header=None, 
                   delimiter=',', 
                   usecols=[3, 4, 5, 7], 
                   names=['qPA', 'pulso', 'resp', 'classe'])

feature_cols = ['qPA', 'pulso', 'resp']

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

if(option == "all"):
   
    random_states = [2,3,9,10,11,14,16,17,19,20,21,24,27,29,33,34,35,43,48]

    #Split data

    csv_file_path = 'CART_trees.csv' 

    fieldnames = ["model_index", "random_state", "max_depth", "criterion", "min_samples_leaf", "accuracy", 
                "precision_class_1", "precision_class_2", "precision_class_3", "precision_class_4", "average precision", 
                "recall_class_1", "recall_class_2", "recall_class_3", "recall_class_4", "average recall", 
                "f_measure_class_1", "f_measure_class_2", "f_measure_class_3", "f_measure_class_4", "average f_measure"]

    model_index = 1

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for random_state in random_states:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state) 
            for max_depth in range(4, 10):
                for criterion in ['gini', 'entropy', 'log_loss']:
                    for min_samples in range(1, 20):
                        # Create Decision Tree classifer object
                        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples, criterion=criterion)
                        # Train Decision Tree Classifer
                        clf = clf.fit(x_train, y_train)
                        #Predict the response for test dataset 
                        y_pred = clf.predict(x_test) 
                        #Calculate metrics
                        accuracy = metrics.accuracy_score(y_test, y_pred)
                        precision = metrics.precision_score(y_test, y_pred, average=None, zero_division=0)
                        recall = metrics.recall_score(y_test,y_pred, average=None, zero_division=0)
                        f_measure = [(2 * p * r)/(p + r) if (p!=0 and r!=0) else 0 for p, r in zip(precision, recall)]
                        average_precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        average_recall = metrics.recall_score(y_test,y_pred, average='weighted', zero_division=0)
                        average_f_measure = metrics.f1_score(y_test, y_pred, average='weighted')
                        #Write to csv
                        tree_infomation = {"model_index" : model_index,
                                        "random_state": random_state,
                                            "max_depth" : max_depth,
                                            "criterion" : criterion,
                                            "min_samples_leaf" : min_samples,
                                            "accuracy" : accuracy,
                                            "precision_class_1": precision[0],
                                            "precision_class_2": precision[1],
                                            "precision_class_3": precision[2],
                                            "precision_class_4": precision[3],
                                            "average precision" : average_precision,
                                            "recall_class_1": recall[0],
                                            "recall_class_2": recall[1],
                                            "recall_class_3": recall[2],
                                            "recall_class_4": recall[3],
                                            "average recall" : average_recall,
                                            "f_measure_class_1": f_measure[0],
                                            "f_measure_class_2": f_measure[1],
                                            "f_measure_class_3": f_measure[2],
                                            "f_measure_class_4": f_measure[3],
                                            "average f_measure" : average_f_measure}
                        writer.writerow(tree_infomation)
                        print(model_index)
                        model_index += 1

    # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # print("Precision:",metrics.precision_score(y_test, y_pred, average=None, zero_division=0))
    # print("Recall Score:",metrics.recall_score(y_test,y_pred, average=None, zero_division=0))
    # print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_pred))
    
    # # Export .dot image
    # PROJECT_ROOT_DIR = "."
    # DATA_DIR = "CART_dot_images"
    # FILE_NAME = "model_" + str(max_depth) + "_" + str(min_samples) + "_" + criterion + ".pkl"

    # def image_path(fig_id):
    #     return os.path.join(PROJECT_ROOT_DIR, DATA_DIR, fig_id)

    # export_graphviz(
    #         clf,
    #         out_file=image_path(f"{FILE_NAME}.dot"),
    #         feature_names=feature_cols,
    #         class_names=['1', '2', '3', '4'],
    #         rounded=True,
    #         filled=True
    # )

    # Source.from_file(image_path(f"{FILE_NAME}.dot"))

    # #Print confirmation and save model

    # print("Nice!")
    # joblib.dump(clf, "CART_models/" +FILE_NAME)

    # Plot ROC curves

    # label_binarizer = LabelBinarizer().fit(y_test)
    # n_classes = 3
    # y_onehot_test = label_binarizer.transform(y_test)

    # for elem in y_test:
    #     if elem == 4:
    #         print("Nice!")

    # y_score = clf.predict_proba(X_test)

    # fig, ax = plt.subplots(figsize=(6,6))

    # colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    # for class_id, color in zip(range(n_classes), colors):

    #     metrics.RocCurveDisplay.from_predictions(
    #         y_onehot_test[:, class_id],
    #         y_score[:, class_id],
    #         name=f"ROC curve for {class_id + 1}",
    #         color=color,
    #         ax=ax,
    #     )

    # plt.plot([0,1], [0,1], "k--", label="chance level (AUC = 0.5)")
    # plt.axis("square")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("One-vs-Rest ROC curves")
    # plt.legend()
    # plt.show()

elif(option == "selected"):

    choosen_models = joblib.load('CART_models.pkl')

    # Save model
    for model in choosen_models:
        model = model.to_dict(orient='records')[0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=model["random_state"])
        clf = DecisionTreeClassifier(max_depth=model["max_depth"], min_samples_leaf=model["min_samples_leaf"], criterion=model["criterion"])
        clf = clf.fit(x_train, y_train) 
        FILE_NAME = "model_" + str(model["model_index"])  +  ".pkl"
        joblib.dump(clf, "CART_models/" + FILE_NAME)

            # Export .dot image
        PROJECT_ROOT_DIR = "."
        DATA_DIR = "CART_dot_images"
        FILE_NAME = "model_" + str(model["model_index"]) + ".pkl"

        def image_path(fig_id):
            return os.path.join(PROJECT_ROOT_DIR, DATA_DIR, fig_id)

        export_graphviz(
                clf,
                out_file=image_path(f"{FILE_NAME}.dot"),
                feature_names=feature_cols,
                class_names=['1', '2', '3', '4'],
                rounded=True,
                filled=True
        )

        Source.from_file(image_path(f"{FILE_NAME}.dot"))


elif(option == "grid_search"):

    model = DecisionTreeClassifier()

    max_depth = [i for i in range(1,21)]
    min_samples_leaf = [i for i in range(10, 21)]
    criterion = ['entropy', 'gini', 'log_loss']

    grid = dict(max_depth = max_depth, 
                min_samples_leaf = min_samples_leaf, 
                criterion = criterion)

    cvFold = RepeatedKFold(n_splits=10, n_repeats=10)
    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=8, cv=cvFold, scoring=('accuracy', 'f1_weighted'), refit='accuracy')

    searchResults = gridSearch.fit(x, y)

    bestModel = searchResults.best_estimator_

    joblib.dump(bestModel, "CART_models/model_grid.pkl")
