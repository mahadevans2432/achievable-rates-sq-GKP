import numpy as np # type: ignore
from class_polar import *
from helper_funcs import *
from sqGKP_funcs import *

# Z(Wi) calculation for bit-flip error channels 
def estimate_Z_Wi_bit_flip(N, p, samples):
    """
    N: code-length
    p: probability of bit-flip
    samples: no. of runs for monte-carlo sampling
    """
    obj = cl_polar_parallel(N,[],samples)
    L = np.log((1-p)/p)
    y = np.random.binomial(1,p,(N,samples))
    L_in = (1-2*y)*L
    obj.sc_decode(L_in)
    LLRs = obj.LLRs
    log_Z_Wi_list = logaddexp_array(-LLRs/2)
    return np.exp(log_Z_Wi_list).flatten()/samples

# Z(Wi) calculation for dit-flip error channels 
def estimate_Z_Wi_dit_flip(N, p, samples):
    """
    N: code-length
    p: probability vector of length d with p[i] = prob of +i ditflip
    samples: no. of runs for monte-carlo sampling
    """
    d = len(p)
    obj = cl_polar_dit(d,N,[],samples)
    L = np.array([[np.log(p[j]/p[(j-i)%d]) for i in range(d)] for j in range(d)])
    y = np.argmax(np.random.multinomial(1,p,(N,samples)),axis=2)
    L_in = L[y, :].transpose(2,0,1)

    obj.sc_decode(L_in)
    log_Z_Wi_list = scipy.special.logsumexp(scipy.special.logsumexp(-obj.LLRs/2,axis=2),axis=0)
    return np.exp(log_Z_Wi_list).flatten()/(samples*(d-1))

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
