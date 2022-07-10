from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

cancer = arff.loadarff(r'../data/breast.w.arff')
df = pd.DataFrame(cancer[0])
df.dropna(inplace=True)

df = df.replace(df['Class'][0], 0)
x = 0
while df['Class'][x] == 0:
    x += 1
df = df.replace(df['Class'][x],1)

X_train,X_test,y_train,y_test=train_test_split(df, df['Class'], random_state=24, test_size=0.1)

error_rate = []
for k in range(1, 433):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred != y_test))
    print("Accuracy:",metrics.accuracy_score(y_test,y_pred), "for k=", k)
    print("Error:", np.mean(y_pred != y_test))

print("Min error:", min(error_rate), "at K =", error_rate.index(min(error_rate))+1)
