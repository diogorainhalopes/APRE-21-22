#%%
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor


k = arff.loadarff(r'../data/kin8nm.arff')
df = pd.DataFrame(k[0])
df.dropna(inplace=True)


x = df[['theta1','theta2','theta3','theta4','theta5','theta6','theta7','theta8']]
y = df['y']


cv_clf=KFold(n_splits=5,random_state=0,shuffle=True)

clf1 = MLPRegressor(hidden_layer_sizes = (3, 2, 10, 10, 10, 10), max_iter=2000,random_state = 24 ,alpha = 0.1)
clf2 = MLPRegressor(hidden_layer_sizes = (3, 2, 10, 10, 10, 10), max_iter=2000, random_state = 24, alpha = 0)

y_pred1 = cross_val_predict(clf1, x, y, cv=cv_clf)
y_pred2 = cross_val_predict(clf2, x, y, cv=cv_clf)
print(y_pred1)

res1 = []
res2 = []
a = y_pred1.tolist()
b = y_pred2.tolist()
for i, j in zip(a, b):
    res1.append( y[a.index(i)] - i) 
    res2.append( y[b.index(j)] - j) 

fig, ax0 = plt.subplots()
ax0.boxplot([res1,res2])
plt.title("Residue Boxplot")
plt.xticks([1,2],["w/ Regularization","w/o Regularization"])
plt.ylabel("Residues")
plt.show()
