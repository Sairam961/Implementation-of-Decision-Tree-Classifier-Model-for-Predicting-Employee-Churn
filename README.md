# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess data

2.Split data into train and test sets

3.Train the decision tree model

4.Test and visualize the results 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: R.Sairam
RegisterNumber:  25000694
*/
```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt

data = pd.read_csv("Employee.csv")

data.columns = data.columns.str.strip()

data = pd.get_dummies(data, columns=['Departments', 'salary'], drop_first=True)

X = data.drop('left', axis=1)

y = data['left'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier( max_depth=5,min_samples_split=20, min_samples_leaf=10,max_features='sqrt',random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))

print('Classification Report:', classification_report(y_test, y_pred))

plt.figure(figsize=(20,10))

plot_tree(clf, feature_names=X.columns, class_names=['Stay','Leave'], filled=True,max_depth=3,fontsize=6)

plt.show()


## Output:
![decision tree classifier model](sam.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
