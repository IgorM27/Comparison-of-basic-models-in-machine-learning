import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

pd.options.mode.chained_assignment=None

data = pd.read_csv("Titanic-Dataset.csv")

columns_target = ['Survived']
columns_train = ['Pclass', 'Sex', 'Age', 'Fare']

X=data[columns_train]
Y=data[columns_target]

X['Age'] = X['Age'].fillna(X['Age'].mean())
d={'male':0, 'female':1}
X['Sex']=X['Sex'].apply(lambda x: d[x])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, random_state=42)

modelSVM = svm.LinearSVC()
modelSVM.fit(X_train,Y_train)

clf = tree.DecisionTreeClassifier(max_depth=10, random_state=21)

bagging = BaggingClassifier(estimator=tree.DecisionTreeClassifier(max_depth=5, random_state=21), n_estimators=100)
bagging.fit(X_train, Y_train)

model = AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(max_depth=1, random_state=21), n_estimators=75)
model.fit(X_train,Y_train)

clf.fit(X_train, Y_train)

modelForest = RandomForestClassifier(n_estimators=150, max_depth=7)
modelForest.fit(X_train,Y_train)

file = open('result.txt', 'w')

file.write("SVM score: " + str(modelSVM.score(X_test, Y_test)))
file.write("\nDecisionTree score: " + str(clf.score(X_test, Y_test)))
file.write("\nBagging with DecisionTree score: " + str(bagging.score(X_test, Y_test)))
file.write("\nRandom forest score: " + str(modelForest.score(X_test, Y_test)))
file.write("\nBoosting with DecisionTree score: " + str(model.score(X_test, Y_test)))