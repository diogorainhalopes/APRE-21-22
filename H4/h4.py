from math import sqrt
import math
import numpy as np

#E-step
def lh(x, u, sigma):
    x = np.array(x)
    u = np.array(u)
    sigma = np.array(sigma)
    res = 1/(2.0*math.pi*sqrt(np.linalg.det(sigma)))
    return res*math.exp(-0.5 * np.matmul(np.matmul(np.transpose(x - u), np.linalg.inv(sigma)), x - u))

def joint(x, u, sigma, p):
    return p * lh(x, u, sigma)

def clusterP(x, u, sigma, p):
    u1 = [[2], [4]]
    u2 = [[-1], [-4]]
    sigma1 = [[1, 0], [0, 1]]
    sigma2 = [[2, 0], [0, 2]]
    p1 = 0.7
    p2 = 0.3
    return joint(x, u, sigma, p)/(joint(x, u1, sigma1, p1) + joint(x, u2, sigma2, p2))

def clusterP2(x, u, sigma, p):
    u1 = mean([[2], [4]], [[1,0],[0,1]], 0.7) 
    u2 = mean([[-1], [-4]], [[2,0],[0,2]], 0.3) 
    sigma1 = var([[2], [4]], [[1,0],[0,1]], 0.7)     
    sigma2 = var([[-1], [-4]], [[2,0],[0,2]], 0.3)   
    p1 = newP([[2], [4]], [[1,0],[0,1]], 0.7)   
    p2 = newP([[-1], [-4]], [[2,0],[0,2]], 0.3)   
    return joint(x, u, sigma, p)/(joint(x, u1, sigma1, p1) + joint(x, u2, sigma2, p2))


#M-step


def mean(u, sigma, p):
    X = [
        [[2], [4]],
        [[-1], [-4]],
        [[-1], [2]],
        [[4], [0]]
    ]
    res = np.array([[0.0], [0.0]])
    for x in X:
        res += clusterP(x, u, sigma, p) * np.array(x)

    res2 = 0.0
    for x in X:
        res2 += clusterP(x, u, sigma, p) 

    return res / res2

def var(u, sigma, p):
    X = [
        [[2], [4]],
        [[-1], [-4]],
        [[-1], [2]],
        [[4], [0]]
    ]
    res = np.array([[0.0, 0.0], [0.0, 0.0]])
    for x in X:
        res += clusterP(x, u, sigma, p) * np.dot((np.array(x) - mean(u, sigma, p)), np.transpose((np.array(x) - mean(u, sigma, p))))

    res2 = 0.0
    for x in X:
        res2 += clusterP(x, u, sigma, p) 

    return (res / res2)



def newP(u, sigma, p):
    X = [
        [[2], [4]],
        [[-1], [-4]],
        [[-1], [2]],
        [[4], [0]]
    ]
    u1 = [[2], [4]]
    u2 = [[-1], [-4]]
    sigma1 = [[1, 0], [0, 1]]
    sigma2 = [[2, 0], [0, 2]]
    p1 = 0.7
    p2 = 0.3
    res = 0.0
    for x in X:
        res += clusterP(x, u, sigma, p)

    res2 = 0.0
    for x in X:
        for c in range(1, 3):
            if c == 1:
                res2 += clusterP(x, u1, sigma1, p1) 
            else:
                res2 += clusterP(x, u2, sigma2, p2)
    return res / res2


###### 2 ######

def a(x, nc, inc):
    if (nc == 1): return 0
    div = 1/(nc -1)
    res = 0
    x = np.array(x)
    for i in inc:
        i = np.array(i)
        res += (np.linalg.norm(x - i))
    return div*(res)

def b(x, nc, outc):
    div = 1/(nc)
    res = 0
    x = np.array(x)
    for i in outc:
        i = np.array(i)
        res += (np.linalg.norm(x - i))
    return div*(res)

def silP(x, inc, outc):
    return (b(x, len(outc), outc) - a(x, len(inc), inc)) / max(a(x, len(inc), inc), b(x, len(outc), outc))

def silC(inc, outc):
    res = 0
    for x in inc:
        res += silP(x, inc, outc)
    return res/ len(inc)


def sil(c1, c2):
    print(silC(c1, c2),'\n' ,silC(c2, c1))
    return (silC(c1, c2) + silC(c2, c1)) /2

