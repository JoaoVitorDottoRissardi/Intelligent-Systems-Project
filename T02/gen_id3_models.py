import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from graphviz import Source
from sklearn.tree import export_graphviz
import os
import sys
import joblib 

#Get parameters from user
max_depth = int(sys.argv[1])
min_samples = int(sys.argv[2])
criterion = sys.argv[3]

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
   
#Generate models with a lot of different characteristics

# models = []
# accuracys = []
# n_of_models = 100
# max_depth_values = list(chain.from_iterable(repeat(x, 10) for x in range(1, 11, 1)))
# min_samples_split_values = range(3, 14, 1)
# print(min_samples_split_values)

# for i in range(n_of_models):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
#     model = DecisionTreeClassifier(max_depth=max_depth_values[i], min_samples_split=min_samples_split_values[i%10])
#     model.fit(X_train, y_train)
#     models.append(model)
#     y_pred = model.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     print("Index: ", i)
#     print("Max depth", max_depth_values[i])
#     print("Min sample split", min_samples_split_values[i%10])
#     print("Accuracy", accuracy)
#     accuracys.append(accuracy)


#Choose best model

# best_model = None
# best_accuracy = 0.0

# for model in models:
#     y_pred = model.predict(X_test)
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = model

# joblib.dump(best_model, 'best_model.pkl')

#Split data until get a split with excalty 1 of 2 gravity 4 victims
number_of_fours = 0

while number_of_fours != 1:

    number_of_fours = 0

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43) 

    for value in y_train:
        if value == 4:
            number_of_fours += 1
    

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples, criterion=criterion)  
 
# # Train Decision Tree Classifer
clf = clf.fit(x_train,y_train) 
 
# #Predict the response for test dataset 
y_pred = clf.predict(x_test) 

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average=None))
print("Recall Score:",metrics.recall_score(y_test,y_pred, average=None, zero_division=0))
print("Confusion Matrix:\n",metrics.confusion_matrix(y_test,y_pred))
 
# Export .dot image
PROJECT_ROOT_DIR = "."
DATA_DIR = "id3_dot_images"
FILE_NAME = "model_" + str(max_depth) + "_" + str(min_samples) + "_" + criterion + ".pkl"

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

#Print confirmation and save model
if number_of_fours == 1:
    print("Nice!")
    joblib.dump(clf, "id3_models/" +FILE_NAME)

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
