#%%
from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

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

train_accuracy_mean = []
test_accuracy_mean = []

for i in [1, 3, 5, 9]:
    x_new = SelectKBest(chi2, k=i).fit_transform(x, y)

    cv_clf=KFold(n_splits=10,random_state=24,shuffle=True)

    clf = DecisionTreeClassifier()
    cv_results = cross_validate(clf, x_new, y, cv=cv_clf, scoring='accuracy', return_train_score=True)

    train_accuracy = cv_results["train_score"]
    test_accuracy = cv_results["test_score"]

    train_accuracy_mean.append(train_accuracy.mean())
    test_accuracy_mean.append(test_accuracy.mean())

X = np.array([1, 3, 5, 9])
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, train_accuracy_mean, color = 'b', width = 0.25)
ax.bar(X + 0.25, test_accuracy_mean, color = 'g', width = 0.25)
ax.set_xlabel('#Selected Features')
ax.set_xticks(X)

ax.set_ylabel('Accuracy')
ax.set_title('Accuracies by #Selected Features')
ax.legend(labels=['Train Accuracy', 'Test Accuracy'], loc=4, shadow=True)

plt.show()
#%%