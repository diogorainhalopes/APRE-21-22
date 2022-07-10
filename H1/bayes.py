#%%
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


cancer = arff.loadarff(r'../data/breast.w.arff')
df = pd.DataFrame(cancer[0])
df.dropna(inplace=True)

df = df.replace(df['Class'][0], 0)
x = 0
while df['Class'][x] == 0:
    x += 1
df = df.replace(df['Class'][x],1)

X_train, X_test, y_train, y_test = train_test_split(df, df.Class, random_state = 24, test_size = 0.1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
#%%