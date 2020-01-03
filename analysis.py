import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

def PrintAccuracy(cmat, y_test, pred):
   # separate out the confusion matrix components
   tpos = cmat[0][0]
   fneg = cmat[1][1]
   fpos = cmat[0][1]
   tneg = cmat[1][0]
   recallScore = round(recall_score(y_test, pred), 2)
   # calculate and display metrics
   print(cmat)
   print( 'Accuracy: '+ str(np.round(100*float(tpos+fneg)/float(tpos+fneg + fpos + tneg),2))+'%')
   print("Sensitivity/Recall for Model/correctly predicted positive observations to the all observations : {recall_score}".format(recall_score = recallScore))

def RunModel(model, X_train, y_train, X_test, y_test):
   model.fit(X_train, y_train.values.ravel())
   pred = model.predict(X_test)
   matrix = confusion_matrix(y_test, pred)
   return matrix, pred


df = pd.read_csv('creditcard.csv')
class_names = {0:'Not Fraud', 1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))

feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:1, 30: ].columns

data_features = df[feature_names]
data_target = df[target]

from sklearn.model_selection import train_test_split
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(data_features,    data_target, train_size=0.70, test_size=0.30, random_state=1)

print("\n\nAnalysis using Logistic Regression : \n")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
PrintAccuracy(cmat, y_test, pred)

print("Analysis using Decision Tree Classifier : \n")
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cmat, pred = RunModel(dt, X_train, y_train, X_test, y_test)
PrintAccuracy(cmat, y_test, pred)

print("Analysis using Naive Bayes : \n")
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
cmat, pred = RunModel(nb, X_train, y_train, X_test, y_test)
PrintAccuracy(cmat, y_test, pred)

print("Analysis using K Nearest Neighbours : \n")
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
cmat, pred = RunModel(knn, X_train, y_train, X_test, y_test)
PrintAccuracy(cmat, y_test, pred)

print("Analysis using Random Forest Classifier : \n")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, n_jobs =4)
cmat, pred = RunModel(rf, X_train, y_train, X_test, y_test)
PrintAccuracy(cmat, y_test, pred)

print("Analysis using Gradient Boosting Classifier : \n")
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(n_estimators=100,max_depth=5)
cmat, pred = RunModel(gb, X_train, y_train, X_test, y_test)
PrintAccuracy(cmat, y_test, pred)
