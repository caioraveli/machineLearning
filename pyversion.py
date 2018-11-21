import pandas as pd
import numpy as np
import tkinter as tk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold #For K-fold cross validation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
import  seaborn as sns


def classification_model(model, data, predictors, outcome):
  #Acerta o modelo:
  model.fit(data[predictors],data[outcome])

  #Faz previsões nos dados de treino:
  predictions = model.predict(data[predictors])

  #Mostra a acurácia
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Performa validação cruzada k-fold com 5 folds
  kf = KFold(n_splits=data.shape[0], shuffle=False)
  error = []
  for train, test in kf.split(data):
    # Filtra dados de treino
    train_predictors = (data[predictors].iloc[train,:])

    # O alvo que estamos usando para treinar o algoritmo
    train_target =  data[outcome].iloc[train]

    # Treinando o algoritmo com previsores e alvo
    model.fit(train_predictors, train_target)

    #Grava erros de cada loop de validação cruzada
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Acerta de novo o modelo para que possa se referir fora da função
  model.fit(data[predictors],data[outcome])



file = pd.DataFrame(pd.read_csv('haberman.csv'), columns=['Age', 'Op_Year','axil_nodes_det','Surv_status'])

print(file)
file.plot()
plt.show()

surv = file['Surv_status'].value_counts()

print(surv)

sns.countplot(x='Surv_status', data=file, palette='hls')
plt.show()


outcome_var = 'Surv_status'
model = LogisticRegression(solver='lbfgs')
predictor_var = ['Age','Op_Year','axil_nodes_det']
classification_model(model, file,predictor_var,outcome_var)


survivers = len(file[file['Surv_status']==1])
nonsurvivers = len(file[file['Surv_status']==2])
perc_survivers = survivers/(survivers+nonsurvivers)
print("Percentual que irão sobreviver após 5 anos: ", perc_survivers*100)
perc_nonsurvivers = nonsurvivers/(survivers+nonsurvivers)
print("Percentual dos que não irão sobreviver: ", perc_nonsurvivers*100)

labels = ['Sobreviventes', 'Não Sobreviventes']
sizes = [survivers,nonsurvivers]
colors = ['lightskyblue', 'red']


plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()








