from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../hospital_deaths_train.csv')

import preprocessor
p = preprocessor.Preprocessor()
p.fit(df)
df = p.transform(df)

Y = df['In-hospital_death']
X = df.drop("In-hospital_death", axis=1)
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)


## KNN

from sklearn.neighbors import KNeighborsClassifier
#tuning
param_grid = {'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']}

knn = KNeighborsClassifier()
#cross-validation
grid_search_knn = GridSearchCV(knn, param_grid, cv=5)
grid_search_knn.fit(X_Train, Y_Train)
knn_score = grid_search_knn.best_score_

print("Best hyperparameters: ", grid_search_knn.best_params_)
print("KNN Accuracy score: ", knn_score)


#metrics
grid_search_knn.fit(X_Train, Y_Train)
Y_Pred = grid_search_knn.predict(X_Test)

accuracy_0 = accuracy_score(Y_Test, Y_Pred)
sensitivity_0 = recall_score(Y_Test, Y_Pred)
specificity_0 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_0 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_0)
print("Sensitivity: ", sensitivity_0)
print("Specificity: ", specificity_0)
print("AUC score: ", auc_score_0)

## Logistic Regression

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_Train, Y_Train)
logistic_score = clf.score(X_Test, Y_Test)
print("Logistic Regression Accuracy score: ",logistic_score)

#metrics
Y_Pred = clf.predict(X_Test)

accuracy_1 = accuracy_score(Y_Test, Y_Pred)
sensitivity_1 = recall_score(Y_Test, Y_Pred)
specificity_1 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_1 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_1)
print("Sensitivity: ", sensitivity_1)
print("Specificity: ", specificity_1)
print("AUC score: ", auc_score_1)

## Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_Train, Y_Train)
gnb_score = gnb.score(X_Test, Y_Test)
print("Gaussian Naive Bayes Accuracy score: ",gnb_score)

#metrics
Y_Pred = gnb.predict(X_Test)

accuracy_2 = accuracy_score(Y_Test, Y_Pred)
sensitivity_2 = recall_score(Y_Test, Y_Pred)
specificity_2 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_2 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_2)
print("Sensitivity: ", sensitivity_2)
print("Specificity: ", specificity_2)
print("AUC score: ", auc_score_2)

## LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_Train, Y_Train)
lda_score = lda.score(X_Test, Y_Test)
print("LDA Accuracy score: ",lda_score)

#metrics
Y_Pred = lda.predict(X_Test)

accuracy_3 = accuracy_score(Y_Test, Y_Pred)
sensitivity_3 = recall_score(Y_Test, Y_Pred)
specificity_3 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_3 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_3)
print("Sensitivity: ", sensitivity_3)
print("Specificity: ", specificity_3)
print("AUC score: ", auc_score_3)

## QDA

from sklearn.discriminant_analysis  import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_Train, Y_Train)
qda_score = qda.score(X_Test, Y_Test)
print("QDA Accuracy score: ",qda_score)

#metrics
Y_Pred = qda.predict(X_Test)

accuracy_4 = accuracy_score(Y_Test, Y_Pred)
sensitivity_4  = recall_score(Y_Test, Y_Pred)
specificity_4 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_4 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_4)
print("Sensitivity: ", sensitivity_4)
print("Specificity: ", specificity_4)
print("AUC score: ", auc_score_4)

## SVM

from sklearn.svm import SVC

#tuning
param_grid = {'C': [0.1, 1, 10],'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],'degree': [3, 5, 7],'gamma': [0.1, 1, 10]}
svm = SVC()
#cross-validation
grid_search_svm = GridSearchCV(svm, param_grid, cv=5)
grid_search_svm.fit(X_Train, Y_Train)
svm_score = grid_search_svm.best_score_

print("Best hyperparameters for SVM: ", grid_search_svm.best_params_)
print("Accuracy score for SVM: ", svm_score)


#metrics
Y_Pred =grid_search_svm.predict(X_Test)

accuracy_5 = accuracy_score(Y_Test, Y_Pred)
sensitivity_5 = recall_score(Y_Test, Y_Pred)
specificity_5= recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_5 = roc_auc = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_5)
print("Sensitivity: ", sensitivity_5)
print("Specificity: ", specificity_5)
print("AUC score: ", auc_score_5)

## Decision Tree

from sklearn.tree import DecisionTreeClassifier

# tuning
param_grid = {'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]}

dt = DecisionTreeClassifier(random_state=0)

# cross-validation
grid_search_dt = GridSearchCV(dt, param_grid, cv=5)
grid_search_dt.fit(X_Train, Y_Train)
dt_score = grid_search_dt.best_score_

print("Best hyperparameters for Decision Tree: ", grid_search_dt.best_params_)
print("Accuracy score for Decision Tree: ", dt_score)


#metrics
Y_Pred =grid_search_dt.predict(X_Test)

accuracy_6 = accuracy_score(Y_Test, Y_Pred)
sensitivity_6 = recall_score(Y_Test, Y_Pred)
specificity_6 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_6 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_6)
print("Sensitivity: ", sensitivity_6)
print("Specificity: ", specificity_6)
print("AUC score: ", auc_score_6)

## Random Forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_Train, Y_Train)
rf_score = rf.score(X_Test, Y_Test)
print("Random Forest Accuracy score: ",rf_score)

#metrics
Y_Pred =rf.predict(X_Test)

accuracy_7 = accuracy_score(Y_Test, Y_Pred)
sensitivity_7 = recall_score(Y_Test, Y_Pred)
specificity_7 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_7 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_7)
print("Sensitivity: ", sensitivity_7)
print("Specificity: ", specificity_7)
print("AUC score: ", auc_score_7)

## Adaboost

from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier()
ab.fit(X_Train, Y_Train)
ab_score= ab.score(X_Test, Y_Test)
print("Adaboost Accuracy score: ",ab_score)

#metrics
Y_Pred =ab.predict(X_Test)

accuracy_8 = accuracy_score(Y_Test, Y_Pred)
sensitivity_8 = recall_score(Y_Test, Y_Pred)
specificity_8 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_8 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_8)
print("Sensitivity: ", sensitivity_8)
print("Specificity: ", specificity_8)
print("AUC score: ", auc_score_8)

# Gradient Descent Boosting

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_Train, Y_Train)
gbc_score = gbc.score(X_Test, Y_Test)
print("Gradient Decsent Boosting Accuracy score: ",gbc_score)

#metrics
Y_Pred =gbc.predict(X_Test)

accuracy_9= accuracy_score(Y_Test, Y_Pred)
sensitivity_9 = recall_score(Y_Test, Y_Pred)
specificity_9 = recall_score(Y_Test, Y_Pred, pos_label=0)
auc_score_9 = roc_auc_score(Y_Test, Y_Pred)

print("Accuracy: ", accuracy_9)
print("Sensitivity: ", sensitivity_9)
print("Specificity: ", specificity_9)
print("AUC score: ", auc_score_9)

## Creating a table of Metrics for all Models

metrics = pd.DataFrame(index=["KNN","Logistic","Naive Bayes","LDA","QDA","SVM","Decision Tree","Random Forest","Adaboost","Gradient Boosting"],
                  columns=["Accuracy","Sensitivity","Specificity","AUC"])

# filling the metrics
metrics.loc["KNN"] = [accuracy_0,sensitivity_0, specificity_0, auc_score_0]
metrics.loc["Logistic"] = [accuracy_1,sensitivity_1, specificity_1, auc_score_1]
metrics.loc["Naive Bayes"] = [accuracy_2,sensitivity_2, specificity_2, auc_score_2]
metrics.loc["LDA"] = [accuracy_3,sensitivity_3, specificity_3, auc_score_3]
metrics.loc["QDA"] = [accuracy_4,sensitivity_4, specificity_4, auc_score_4]
metrics.loc["SVM"] = [accuracy_5,sensitivity_5, specificity_5, auc_score_5]
metrics.loc["Decision Tree"] = [accuracy_6,sensitivity_6, specificity_6, auc_score_6]
metrics.loc["Random Forest"] = [accuracy_7,sensitivity_7, specificity_7, auc_score_7]
metrics.loc["Adaboost"] = [accuracy_8,sensitivity_8, specificity_8, auc_score_8]
metrics.loc["Gradient Boosting"] = [accuracy_9,sensitivity_9, specificity_9, auc_score_9]

print(metrics)


# create a table plot
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
ax.table(cellText=metrics.values, colLabels=metrics.columns, rowLabels=metrics.index, loc='center')
plt.savefig("classifier_performance.png")