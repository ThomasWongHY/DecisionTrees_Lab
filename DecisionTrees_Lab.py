import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot as plt

# Load & check the data
data_HoYin = pd.read_csv('/content/student-por.csv', sep=';')

print(data_HoYin.info(),'\n')
print(data_HoYin.isna().sum(),'\n')
print(data_HoYin.describe(),'\n')
for i in data_HoYin.select_dtypes(include=object):
    print(i)
    print(data_HoYin[i].unique())
    print()

data_HoYin['pass_HoYin'] = np.where(data_HoYin['G1'] + data_HoYin['G2'] + data_HoYin['G3'] >= 35, 1, 0)
data_HoYin.drop(['G1','G2','G3'], axis=1, inplace=True)

features_HoYin = data_HoYin.drop(['pass_HoYin'], axis=1)
target_HoYin = data_HoYin['pass_HoYin']
print(target_HoYin.value_counts())

features_HoYin.shape

numeric_features_HoYin = features_HoYin.select_dtypes(exclude=object).columns
cat_features_HoYin = features_HoYin.select_dtypes(include=object).columns

transformer_HoYin = ColumnTransformer([
    ("cat", OneHotEncoder(), cat_features_HoYin),
    ("num", StandardScaler(), numeric_features_HoYin)
])

clf_HoYin = DecisionTreeClassifier(criterion="entropy", max_depth=5)

pipeline_HoYin = Pipeline([
    ('transformer', transformer_HoYin),
    ('clf_HoYin', clf_HoYin)
])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=44)
for train_index, test_index in sss.split(features_HoYin, target_HoYin):
    X_train_HoYin = features_HoYin.loc[train_index]
    X_test_HoYin = features_HoYin.loc[test_index]
    y_train_HoYin = target_HoYin.loc[train_index]
    y_test_HoYin = target_HoYin.loc[test_index]

# Build Classification Models
data_HoYin_pipe = pipeline_HoYin.fit(X_train_HoYin, y_train_HoYin)

cv = KFold(n_splits=10, shuffle=True, random_state=44)
scores = cross_val_score(pipeline_HoYin, X_train_HoYin, y_train_HoYin, cv=cv)

print(scores)
print(scores.mean())

import graphviz

feature_name = list(pipeline_HoYin.named_steps['transformer'].named_transformers_['cat'].get_feature_names_out(cat_features_HoYin))\
                + list(pipeline_HoYin.named_steps['transformer'].named_transformers_['num'].get_feature_names_out(numeric_features_HoYin))

dot_data = tree.export_graphviz(clf_HoYin, feature_names=feature_name, 
                                class_names=target_HoYin.name, filled=True, 
                                rounded=True, special_characters=True
                                )
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('decision_tree',view=True)
graph

print("Training accuracy", pipeline_HoYin.score(X_train_HoYin, y_train_HoYin))
print("Testing accuracy", pipeline_HoYin.score(X_test_HoYin, y_test_HoYin))

y_pred = pipeline_HoYin.predict(X_test_HoYin)
print("accuracy", accuracy_score(y_test_HoYin, y_pred))
print("precision",precision_score(y_test_HoYin, y_pred))
print("recall",recall_score(y_test_HoYin, y_pred))
print(confusion_matrix(y_test_HoYin, y_pred))

# Fine tune the model
parameters = {"clf_HoYin__min_samples_split": range(10, 300, 20),
              "clf_HoYin__max_depth": range(1, 30, 2),
              "clf_HoYin__min_samples_leaf": range(1, 15, 3)}

clf_HoYin = RandomizedSearchCV(
    estimator=pipeline_HoYin, 
    param_distributions=parameters, 
    scoring='accuracy', cv=5,
    n_iter=7, refit=True, 
    verbose=3)

search = clf_HoYin.fit(X_train_HoYin, y_train_HoYin)

best_model = search.best_estimator_
print("Best Params:", search.best_params_)
print("Best Score:",search.best_score_)
print("Best Estimator:",best_model)

ypred_grid = best_model.predict(X_test_HoYin)
print("accuracy", accuracy_score(y_test_HoYin, ypred_grid))
print("precison",precision_score(y_test_HoYin, ypred_grid))
print("recall",recall_score(y_test_HoYin, ypred_grid))

import joblib

model_file = 'best_model_HoYin.pkl'
joblib.dump(best_model, model_file)

pipe_file = 'pipeline_HoYin.pkl'
joblib.dump(pipeline_HoYin, pipe_file)