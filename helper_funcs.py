import numpy as np # type: ignore
import scipy # type: ignore


def cutoff_set_gen(cutoff, N, Z_X, Z_P):
    setI = []
    setA = []
    setP = []
    setE = []
    for i in range(N):
        if (Z_X[i] <= cutoff and Z_P[i] <= cutoff):
            setI.append(i)
        elif (Z_X[i] <= cutoff and Z_P[i] > cutoff):
            setP.append(i)
        elif (Z_X[i] > cutoff and Z_P[i] <= cutoff):
            setA.append(i)
        else:
            setE.append(i)
    print(len(setI),len(setA),len(setP),len(setE))
    setA = np.array(setA)
    setP = np.array(setP)
    setI = np.array(setI)
    setE = np.array(setE)
    return setI, setA, setP, setE

def capacity_set_gen(N,cap,Z_X,Z_P):
    K = int(cap*N)
    I = int(N*(cap+1)/2)
    goodA = np.argsort(Z_X)[:I]
    badA = np.argsort(Z_X)[I:]
    goodP = np.argsort(Z_P)[:I]
    badP = np.argsort(Z_P)[I:]
    setI = np.intersect1d(goodA,goodP)
    setE = np.intersect1d(badA,badP)
    setA = np.setdiff1d(badA,setE)
    setP = np.setdiff1d(badP,setE)
    return setI, setA, setP, setE

def capacity_set_gen_min_dist(N,cap,Z_X,Z_P,min_d):
    K = int(cap*N)
    I = int(N*(cap+1)/2)
    weightsA = np.array([2**np.binary_repr(i).count('1') for i in range(N)])
    weightsP = np.flip(weightsA)
    
    goodA = np.argsort(Z_X)[:I]
    low_weight = []
    goodA_temp = []
    badA = np.argsort(Z_X)[I:]
    for j in goodA:
        if weightsA[j] < min_d:
            low_weight.append(j)
        else:
            goodA_temp.append(j)
    goodA = np.array(goodA_temp)
    badA = np.append(badA,low_weight)

    goodP = np.argsort(Z_P)[:I]
    badP = np.argsort(Z_P)[I:]
    low_weight = []
    goodP_temp = []
    for j in goodP:
        if weightsP[j] < min_d:
            low_weight.append(j)
        else:
            goodP_temp.append(j)
    goodP = np.array(goodP_temp)
    badP = np.append(badP,low_weight)

    setI = np.intersect1d(goodA,goodP)
    setE = np.intersect1d(badA,badP)
    setA = np.setdiff1d(badA,setE)
    setP = np.setdiff1d(badP,setE)
    return setI, setA, setP, setE

def logaddexp_array(A):
    if A.shape[1] == 1:
        return A
    else:
        return np.logaddexp(logaddexp_array(A[:,:A.shape[1]//2]),logaddexp_array(A[:,A.shape[1]//2:]))
    
def worst_case_rate(d,p,Z_X,Z_P):
    N = len(Z_X)
    ptot = 0
    psort = (d-1)*np.sort(Z_X + Z_P)/2
    for i in range(N):
        ptot += psort[i]
        if ptot >= p:
            return (i+1)/N