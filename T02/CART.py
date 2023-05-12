import pandas as pd
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from graphviz import Source
from sklearn.tree import export_graphviz
import os
 

data = pd.read_csv('treino_sinais_vitais_com_label.txt', 
                   header=None, 
                   delimiter=',', 
                   usecols=[3, 4, 5, 7], 
                   names=['qPA', 'pulso', 'resp', 'classe'])

feature_cols = ['qPA', 'pulso', 'resp']

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(X)

print(y)
   
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test
   
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=10)  
 
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train) 
 
#Predict the response for test dataset 
y_pred = clf.predict(X_test) 
 
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
 
# Where to save the figures
PROJECT_ROOT_DIR = "."
DATA_DIR = "dot_images"
FIGURE_ID = "my_decision_tree"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, DATA_DIR, fig_id)

export_graphviz(
        clf,
        out_file=image_path(f"{FIGURE_ID}.dot"),
        feature_names=feature_cols,
        class_names=['1', '2', '3', '4'],
        rounded=True,
        filled=True
    )

Source.from_file(image_path(f"{FIGURE_ID}.dot"))