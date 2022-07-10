import numpy as np
from math import sqrt

def dist(v,j):
    return (sqrt(v[0]**2+v[1]**2+v[2]**2))**j

def line_dist(v):
    res = []
    for j in range(0, 4):
        res=res+[dist(v,j)]
    return res

def matrix_dist(m):
    res=[]
    for i in range(0,8):
        res=res+[line_dist(m[i])]
    return res

def Phi():
    x = np.array([[1,1,0], [1,1,5], [0,2,4], [1,2,3], [2,0,7], [1,1,1], [2,0,2], [0,2,9]])
    return np.array(matrix_dist(x))

def Phi_test():
    x_test = np.array([[2,0,0], [1,2,1]])
    return np.array(matrix_dist(x_test))


def W():
    z = np.array([[1],[3],[2],[0],[6],[4],[5],[7]])
    phi = Phi()
    
    phit = np.transpose(phi)

    mul = np.matmul(phit,phi)

    inv = np.linalg.inv(mul)

    mul2 = np.matmul(inv,phit)

    w = np.matmul(mul2,z)

    return w

def f():
    w = W()
    phi = Phi()

    return np.matmul(phi, w)

def f_test():
    w = W()
    phi2 = Phi_test()

    return np.matmul(phi2, w)

print(Phi())