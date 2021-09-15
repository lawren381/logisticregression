import pandas as pd
from pandas import value_counts
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedKFold, GridSearchCV
import numpy as np
import pickle
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score

# load csv file

df = pd.read_csv("new2.csv")
x = df.drop("target", axis=1)

y = df["target"]
print(y.value_counts() / df.shape[0])

# Split Data into train and test sets


# split into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)

# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# define class weights
w = {1: 4, 0: 96}
# instantiate a model

classifier = LogisticRegression(class_weight=w, random_state=13)

# fit a model

classifier.fit(x_train, y_train)
ypred = classifier.predict(x_test)
print(ypred)

print(f'Accuracy Score: {accuracy_score(y_test, ypred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, ypred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, ypred)}')
print(f'Recall score: {recall_score(y_test, ypred)}')

# make a pickle of the model
pickle.dump(classifier, open("model.pkl", "wb"))
