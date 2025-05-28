import numpy as np
from class_polar import *
from sqGKP_funcs import *

# Z(Wi) calculation for the dit-flip channel with analog output (Q quadrature)
def estimate_Z_Wi_displ_sq_dGKP_Q(d, N, sig, samples,alpha=1):
    """
    d: qudit dimension
    N: code-length
    sig: displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    obj = cl_polar_dit_alpha(d,N,[],samples,alpha)
    
    x_error = np.random.normal(0,sig,(N,samples))
    u = np.array(np.rint(x_error/np.sqrt(2*np.pi/d))%d,dtype=int)
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/d))
    k1[k1>=np.sqrt(1/(4*d))] -= np.sqrt(1/d)

    Lu = np.zeros((d,N,samples))
    for i in range(N):
        for j in range(samples):
            Lu[:,i,j] = pu_belief(k1[i,j],sig,u[i,j],d)

    obj.sc_decode(Lu)
    logZ_Wi_list_X = scipy.special.logsumexp(scipy.special.logsumexp(-obj.LLRs/2,axis=-1),axis=0)

    return (np.exp(logZ_Wi_list_X)).flatten()/((d-1)*samples)

# Z(Wi) calculation for the dit-flip (phase-flip for the qudit) channel with analog output (P quadrature)
def estimate_Z_Wi_displ_sq_dGKP_P(d, N, sig, samples,alpha=1):
    """
    d: qudit dimension
    N: code-length
    sig: displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    obj = cl_polar_dit_alpha(d,N,[],samples,-alpha)
    
    x_error = np.random.normal(0,sig,(N,samples))
    u = np.array(np.rint(x_error/np.sqrt(2*np.pi/d))%d,dtype=int)
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/d))
    k1[k1>=np.sqrt(1/(4*d))] -= np.sqrt(1/d)

    Lu = np.zeros((d,N,samples))
    for i in range(N):
        for j in range(samples):
            Lu[:,i,j] = pu_belief(k1[i,j],sig,u[i,j],d)

    obj.sc_decode(Lu)
    logZ_Wi_list_P = np.flip(scipy.special.logsumexp(scipy.special.logsumexp(-obj.LLRs/2,axis=-1),axis=0))

    return (np.exp(logZ_Wi_list_P)).flatten()/((d-1)*samples)
