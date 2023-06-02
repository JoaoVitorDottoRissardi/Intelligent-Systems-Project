import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

#Load data
data = pd.read_csv('treino_sinais_vitais_com_label.txt', 
                   header=None, 
                   delimiter=',', 
                   usecols=[3, 4, 5, 7], 
                   names=['qPA', 'pulso', 'resp', 'classe'])

feature_cols = ['qPA', 'pulso', 'resp']

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(x)

print(y)
   
#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43) 

csv_file_path = 'id3_trees.csv' 

fieldnames = ["model_index", "max_depth", "criterion", "min_samples_leaf", "accuracy", 
              "precision_class_1", "precision_class_2", "precision_class_3", "precision_class_4", "average precision", 
              "recall_class_1", "recall_class_2", "recall_class_3", "recall_class_4", "average recall", 
              "f_measure_class_1", "f_measure_class_2", "f_measure_class_3", "f_measure_class_4", "average f_measure"]

model_index = 1

# with open(csv_file_path, mode='w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()

for max_depth in range(4, 10):
    for criterion in ['gini', 'entropy', 'log_loss']:
        for min_samples in range(1, 20):
            # Create Decision Tree classifer object
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples, criterion=criterion)
            # Train Decision Tree Classifer
            clf = clf.fit(x_train, y_train)
            #Predict the response for test dataset 
            y_pred = clf.predict(x_test) 
            #Save model
            if model_index == 253 or model_index == 81 or model_index == 7 or model_index == 291:
                FILE_NAME = "model_" + str(max_depth) + "_" + str(min_samples) + "_" + criterion + "_" + str(model_index) +  ".pkl"
                joblib.dump(clf, "id3_models/" + FILE_NAME)
            #Calculate metrics
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average=None, zero_division=0)
            recall = metrics.recall_score(y_test,y_pred, average=None, zero_division=0)
            f_measure = [(2 * p * r)/(p + r) if (p!=0 and r!=0) else 0 for p, r in zip(precision, recall)]
            average_precision = sum(precision) / float(len(precision))
            average_recall = sum(recall) / float(len(recall))
            average_f_measure = sum(f_measure) / float(len(f_measure))
            #Write to csv
            tree_infomation = {"model_index" : model_index,
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
            #writer.writerow(tree_infomation)
            print(model_index)
            model_index += 1

# Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred, average=None, zero_division=0))
# print("Recall Score:",metrics.recall_score(y_test,y_pred, average=None, zero_division=0))
# print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_pred))
 
# # Export .dot image
# PROJECT_ROOT_DIR = "."
# DATA_DIR = "id3_dot_images"
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
# joblib.dump(clf, "id3_models/" +FILE_NAME)

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
