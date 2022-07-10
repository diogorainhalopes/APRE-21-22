#%%
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cancer = arff.loadarff(r'../data/breast.w.arff')
df = pd.DataFrame(cancer[0])
df.dropna(inplace=True)

df = df.replace(df['Class'][0], 0)
x = 0
while df['Class'][x] == 0:
    x += 1
df = df.replace(df['Class'][x],1)

x = df[['Clump_Thickness','Cell_Size_Uniformity','Cell_Shape_Uniformity','Marginal_Adhesion','Single_Epi_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
y = df['Class']


alpha = 0.1
cv_clf=KFold(n_splits=5,random_state=0,shuffle=True)

clf1 = MLPClassifier(hidden_layer_sizes = (3, 2), alpha=alpha, max_iter=2000, random_state = 24)
clf2 = MLPClassifier(hidden_layer_sizes = (3, 2), alpha=alpha, max_iter=2000, random_state = 24, early_stopping= True)


y_pred1 = cross_val_predict(clf1, x, y, cv=cv_clf)
y_pred2 = cross_val_predict(clf2, x, y, cv=cv_clf)

cm1 = confusion_matrix(y, y_pred1)
cm2 = confusion_matrix(y, y_pred2)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)

disp1.plot()
disp2.plot()
plt.show()
#%%
