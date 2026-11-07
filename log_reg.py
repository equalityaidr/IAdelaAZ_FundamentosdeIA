# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:08:44 2020

@author: DELL
"""

#Ejercicio Regresion Logistica o Logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:,:2] #Taking only two features
y = iris.target

df= pd.DataFrame(iris.data, columns=iris.feature_names)

#Data Mining o Mineria de Datos
print("Printing first 10 features set")
print(X[:10,:])
print("--------------------------")
print("Printing first 10 targets")
print(y[:10])
print(iris.DESCR)

from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')

#Cantidad por clase
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))

#Scatter plat o grafico de dispersion
plt.figure(2, figsize=(8, 6))
plt.clf()
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(),loc = "upper right", title="Classes")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#Eliminando Clase '2'
new_y = y[:100]
new_X = X[:100]

#Scatter2 o dispersion 2
plt.figure(2, figsize=(8, 6))
plt.clf()
scatter = plt.scatter(new_X[:, 0], new_X[:, 1], c=new_y,cmap=plt.cm.Set1, edgecolor='k')
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#train test split, separando la data en prueba y entrenamiento
X_train,X_test,Y_train,Y_test=train_test_split(new_X,new_y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

#Logistic Regression o regresion logistica
logreg = LogisticRegression(solver='lbfgs', C=0.12)
logreg.fit(X_train,Y_train)

#Scores, coeficientes, intercepto
print(logreg.score(X_train,Y_train))
print (logreg.intercept_, logreg.coef_)

#Predicciones
predict_y = logreg.predict(X_test)




#Accuracy, metricas de precision
print ('Accuracy from sk-learn: {0}'.format(logreg.score(X_test,Y_test)))

#Gridsearch, metodo para encontrar el mejor valor del parametro C para regresion logistica
parameters = {'C':np.linspace(0.01,1,10)}
clf = GridSearchCV(LogisticRegression(solver='lbfgs'),parameters,cv=5)
clf.fit(X_train,Y_train)

#Best score y parameters, mejor score y parametros
print(clf.best_score_)
print(clf.best_params_)

#Accuracy (tamaño), metricas
print ('Accuracy after GridSearch: {0}'.format(clf.score(X_test,Y_test)))

#Clasificacion Multiclase

#Split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

#Regresion Logistica
logreg_multi = LogisticRegression(solver='lbfgs',multi_class='ovr')
logreg_multi.fit(X_train,Y_train)

#Metricas
print(logreg_multi.score(X_train,Y_train))
print (logreg_multi.intercept_, logreg.coef_)

#prediciendo
predict_y = logreg_multi.predict(X_test)
predict_y

#Accuracy, metricas
print ('Accuracy from sk-learn: {0}'.format(logreg_multi.score(X_test,Y_test)))

# Matriz de Confusion 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predict_y)

#GridSearch
parameters = {'C':np.linspace(0.01,1,10)}
clf_multi = GridSearchCV(LogisticRegression(solver='lbfgs',multi_class='ovr'),parameters,cv=5)
clf_multi.fit(X_train,Y_train)

#Mejor score y params
print(clf_multi.best_score_)
print(clf_multi.best_params_)
print(clf_multi.best_estimator_)

#prediciendo
predict_y =clf_multi.predict(X_test)
predict_y

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predict_y)

#Accuracy GS
print ('Accuracy after GridSearch: {0}'.format(clf_multi.score(X_test,Y_test)))

import pickle

pickle.dump(clf_multi, open('model.pkl', 'wb'))

pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)
