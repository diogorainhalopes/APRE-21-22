#%%
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt


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



kmeans2 = KMeans(n_clusters = 2, init = 'random', random_state = 0).fit(x)
kmeans3 = KMeans(n_clusters = 3, init = 'random', random_state = 0).fit(x)


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

print('ECR K = 2: ' + str(ecr(2, kmeans2.labels_)))
print('ECR K = 3: ' + str(ecr(3, kmeans3.labels_)))
print('sil 2: ' + str(metrics.silhouette_score(x, kmeans2.labels_, metric='euclidean')))
print('sil 3: ' + str(metrics.silhouette_score(x, kmeans3.labels_, metric='euclidean')))
#%%
