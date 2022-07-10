from math import sqrt
import math

def cov(v1,v2):
    media1 = 0
    media2 = 0
    res = 0
    for x in v1:
        media1 = media1 + x
    media1 = media1 / len(v1)
    for x in v2:
        media2 = media2 + x
    media2 = media2 / len(v2)
    for i in range(len(v1)):
        res = res + (v1[i] - media1)*(v2[i] - media2)
    res = res / (len(v1)-1)
    return res


def mean(vec):
    media = 0
    for a in vec:
        media = media + a
    return media / len(vec)


def var(vec):
    media = 0
    res = 0
    for a in vec:
        media = media + a
    media = media / len(vec)
    for i in range(len(vec)):
        res = res + (vec[i] - media)**2
    return res / (len(vec)-1)


def det(vec):
    return vec[0]*vec[3] - vec[1]*vec[2]


def matrix_sub(vec1, vec2):
    res = []
    for i in range(len(vec1)):
        res = res + [(vec1[i] - vec2[i])]
    return res


def simetrica(vec):
    inv = 1 / det(vec)
    return [inv*vec[3], -inv*vec[1], -inv*vec[2], inv*vec[0]]


def mult1(vec1, vec2):
    return [vec1[0]*vec2[0]+vec1[1]*vec2[2], vec1[0]*vec2[1]+vec1[1]*vec2[3]]


def mult2(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]


def norm(vec, x):
    variance = var(vec)
    res = 1/(sqrt(2.0*math.pi)*sqrt(variance))
    return res*math.exp((-1/(2*variance))*((x-mean(vec))**2))


def mat(vec1, vec2, vecx):
    var1 = var(vec1)
    var2 = var(vec2)
    var3 = cov(vec1, vec2)
    media = [mean(vec1), mean(vec2)]
    sigma = [var1, var3, var3, var2]
    print(simetrica(sigma))
    x = matrix_sub(vecx, media)
    res = 1/(2.0*math.pi*sqrt(det(sigma)))
    return res*math.exp(-0.5 * mult2(mult1(x, simetrica(sigma)), x))


def calculate(y1, Py2, y3, y4, y1vec, y3vec, y4vec, PX):
    Py1 = norm(y1vec, y1)
    print(Py1)
    Py3 = mat(y3vec, y4vec, [y3, y4])
    print(Py3)
    return Py1 * Py2 * Py3 * PX