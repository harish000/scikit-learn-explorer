'''
Created on Aug 23, 2016

@author: Sriharish
'''
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from random import randrange

iris = datasets.load_iris()
# iris = np.array(iris)
dataset = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
dataset = dataset.iloc[np.random.permutation(len(dataset))]
dataset.reset_index(inplace=True)
count = len(dataset)
k =randrange(2,10)
count = int(count/k)
test = dataset.iloc[:count,:4]
testActual = dataset.iloc[:count,5:]
train = dataset.iloc[count:,:4]
target = np.ravel(dataset.iloc[count:,5:])
# print(type(testActual))
# print(len(train))
rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)
print(pd.DataFrame(data=np.c_[rf.predict(test),testActual],columns=['Predicted Value']+['Actual Value']))
