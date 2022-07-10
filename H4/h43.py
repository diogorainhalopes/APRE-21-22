#%%
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif as MIC



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

def topF():
    mi_score = MIC(x, y)
    j = mi_score.tolist()
    j.sort()
    return [np.where(mi_score == j[-1])[0][0], np.where(mi_score == j[-2])[0][0]]


def newX():
    ind = topF()
    nx = x.columns.values.tolist()
    return df[[nx[ind[0]], nx[ind[1]]]]

X = newX()
F1 = X.columns.values.tolist()[0]
F2 = X.columns.values.tolist()[1]

kmeans3 = KMeans(n_clusters = 3, init = 'random').fit(X)
centroids = kmeans3.cluster_centers_
c0 = X[kmeans3.predict(X) == 0]
c1 = X[kmeans3.predict(X) == 1]
c2 = X[kmeans3.predict(X) == 2]
fig, ax = plt.subplots()
plt.scatter(c0.loc[:,F1].tolist() , c0.loc[:,F2].tolist(), color = 'red')
plt.scatter(c1.loc[:,F1].tolist() , c1.loc[:,F2].tolist(), color = 'green')
plt.scatter(c2.loc[:,F1].tolist() , c2.loc[:,F2].tolist(), color = 'blue')
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = '0')
ax.legend(labels=['Cluster 0', 'Cluster 1', 'Cluster 2'], loc=2, fontsize = 9)
ax.set_xlabel(F1)
ax.set_ylabel(F2)
ax.set_title('Clustering solution with top-2 features with higher mutual information')
ax.grid(True)
plt.show()


def phi(i, C):
    L = y.tolist()
    cont0 = 0
    cont1 = 0
    for j in range(len(C)):
        if C[j] == i:
            if L[j] == 1:
                cont1 +=1
            else:
                cont0 +=1
    return max(cont0, cont1)
# C = kmeans#.labels_
def ecr(K, C):
    res = 0
    for i in range(K):
        z = 0
        for j in C:
            if i == j:  
                z+=1
        res += z - phi(i, C)
    return (1/K) * (res)

print('ECR: ' + str(ecr(3, kmeans3.labels_)))
print('SIL: ' + str(metrics.silhouette_score(x, kmeans3.labels_, metric='euclidean')))

# %%
