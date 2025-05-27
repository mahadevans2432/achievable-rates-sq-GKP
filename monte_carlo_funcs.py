import numpy as np # type: ignore
from class_polar import *
from helper_funcs import *
from sqGKP_funcs import *

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

# Z(Wi) calculation for qudits 
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

def estimate_Z_Wi_pauli(N, p_uv, samples):
    ###Something is wrong with the phase part FIX IT!!!
    """
    N: code-length
    p_uv: array with 4 entries for probability distribution of bit and phase flips in order [p_00,p_01,p_10,p_11]
    samples: no. of runs for monte-carlo sampling
    """
    tol = 1e-30
    obj = cl_polar_parallel(N,[],samples)
    
    pu = p_uv[2]+p_uv[3]
    p0v = p_uv[1]/(p_uv[1]+p_uv[0]+tol)
    p1v = p_uv[3]/(p_uv[3]+p_uv[2]+tol)

    L0 = np.log((1-p0v)/(p0v+tol))
    L1 = np.log((1-p1v)/(p1v+tol))
    u = np.random.binomial(1,pu,(N,samples)) #Bit error
    v = np.random.binomial(1,p0v*(1-u)+p1v*u,(N,samples)) #Phase error
    
    Lu = (1-2*u)*np.log((1-pu)/pu)
    Lv = (1-2*v)*(L0*(1-u) + L1*u)
        
    obj.sc_decode(Lu)
    log_Z_Wi_list_bit = logaddexp_array(-obj.LLRs/2)

    obj.sc_decode(Lv)
    log_Z_Wi_list_phase = logaddexp_array(-np.flip(obj.LLRs/2,axis=0))

    return np.exp(log_Z_Wi_list_bit).flatten()/samples, np.exp(log_Z_Wi_list_phase).flatten()/samples


def pauli_polar_MC_error(polar_bit, polar_phase, p_uv, samples):
    """
    polar_bit: cl_polar_parallel object with bits frozen in set A and no. of samples set
    polar_phase: cl_polar_parallel object with bits frozen in set P and no. of samples set
    p_uv: array with 4 entries for probability distribution of bit and phase flips in order [p_00,p_01,p_10,p_11]
    samples: no. of runs for monte-carlo sampling
    """
    tol = 1e-30
    N = polar_bit.N
    
    pu = p_uv[2]+p_uv[3]
    p0v = p_uv[1]/(p_uv[1]+p_uv[0]+tol)
    p1v = p_uv[3]/(p_uv[3]+p_uv[2]+tol)

    L0 = np.log((1-p0v)/(p0v+tol))
    L1 = np.log((1-p1v)/(p1v+tol))
    u = np.random.binomial(1,pu,(N,samples)) #Bit error
    v = np.random.binomial(1,p0v*(1-u)+p1v*u,(N,samples)) #Phase error

    Lu = np.log((1-pu)/pu)*np.ones(u.shape)
    u_hat = polar_bit.sc_decode(Lu)
    Lv = np.flip((1-u_hat)*L0 + (u_hat)*L1)
    v_hat = np.flip(polar_phase.sc_decode(Lv))

    avg_u_diff  = np.sum(np.sum((u_hat + u)%2,axis=0))/samples
    avg_v_diff = np.sum(np.sum((v_hat + v)%2,axis=0))/samples
    avg_error_count = np.sum(np.minimum(1,np.sum((u_hat + u)%2,axis=0) + np.sum((v_hat + v)%2,axis=0)))/samples
    
    return avg_error_count, avg_u_diff, avg_v_diff


def estimate_Z_Wi_displ_sqGKP(N, sig, samples):
    """
    N: code-length
    sig: displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    obj = cl_polar_parallel(N,[],samples)
    
    x_error = np.random.normal(0,sig,(N,samples))
    y_error = np.random.normal(0,sig,(N,samples))
    u = np.rint(x_error/np.sqrt(np.pi))%2
    v = np.rint(y_error/np.sqrt(np.pi))%2
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/2))
    k1[k1>=np.sqrt(1/8)] -= np.sqrt(1/2)
    k2 = np.mod(y_error/np.sqrt(2*np.pi),np.sqrt(1/2))
    k2[k2>=np.sqrt(1/8)] -= np.sqrt(1/2)

    pk1 = pu(k1,sig)
    pk2 = pu(k2,sig)
    Lu = (1-2*u)*np.log((1-pk1)/pk1)
    Lv = (1-2*v)*np.log((1-pk2)/pk2)
        
    obj.sc_decode(Lu)
    logZ_Wi_list_X = logaddexp_array(-(obj.LLRs/2))

    obj.sc_decode(Lv)
    logZ_Wi_list_P = logaddexp_array(-np.flip(obj.LLRs/2))

    return (np.exp(logZ_Wi_list_X)).flatten()/samples, (np.exp(logZ_Wi_list_P)).flatten()/samples


def disp_sqGKP_polar_MC_error(polar_bit, polar_phase, sig, samples):
    """
    polar_bit: cl_polar object with bits frozen in set A
    polar_phase: cl_polar object with bits frozen in set P
    sig: spread of displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    N = polar_bit.N
    
    x_error = np.random.normal(0,sig,(N,samples))
    y_error = np.random.normal(0,sig,(N,samples))
    u = np.rint(x_error/np.sqrt(np.pi))%2
    v = np.rint(y_error/np.sqrt(np.pi))%2
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/2))
    k1[k1>=np.sqrt(1/8)] -= np.sqrt(1/2)
    k2 = np.mod(y_error/np.sqrt(2*np.pi),np.sqrt(1/2))
    k2[k2>=np.sqrt(1/8)] -= np.sqrt(1/2)
        
    pk1 = pu(k1,sig)
    pk2 = pu(k2,sig)
    Lu = np.log((1-pk1)/pk1)
    Lv = np.flip(np.log((1-pk2)/pk2),axis=0)
        
    polar_bit.uAc = polar_bit._encode(u) #this is u'
    polar_phase.uAc = (polar_phase._encode(np.flip(v,axis=0))) #this is flipped v'

    u_hat = polar_bit.sc_decode(Lu)
    v_hat = np.flip(polar_phase.sc_decode(Lv),axis=0)
    
    avg_u_diff  = np.sum(np.sum((u_hat + u)%2,axis=0))/samples
    avg_v_diff = np.sum(np.sum((v_hat + v)%2,axis=0))/samples
    avg_error_count = np.sum(np.minimum(1,np.sum((u_hat + u)%2,axis=0) + np.sum((v_hat + v)%2,axis=0)))/samples
    
    return avg_error_count, avg_u_diff, avg_v_diff

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

def est_all_Z_Wi_displ_sq_dGKP(d, N, sig, samples):
    """
    d: qudit dimension
    N: code-length
    sig: displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    obj = cl_polar_dit(d,N,[],samples)
    
    x_error = np.random.normal(0,sig,(N,samples))
    u = np.array(np.rint(x_error/np.sqrt(2*np.pi/d))%d,dtype=int)
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/d))
    k1[k1>=np.sqrt(1/(4*d))] -= np.sqrt(1/d)

    Lu = np.zeros((d,N,samples))
    for i in range(N):
        for j in range(samples):
            Lu[:,i,j] = pu_belief(k1[i,j],sig,u[i,j],d)

    obj.sc_decode(Lu)
    logZ_Wi_list_X = scipy.special.logsumexp(-obj.LLRs/2,axis=-1)

    return (np.exp(logZ_Wi_list_X))/(samples)

def est_worst_Z_Wi_displ_sq_dGKP(d, N, sig, samples):
    """
    d: qudit dimension
    N: code-length
    sig: displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    obj = cl_polar_dit(d,N,[],samples)
    
    x_error = np.random.normal(0,sig,(N,samples))
    u = np.array(np.rint(x_error/np.sqrt(2*np.pi/d))%d,dtype=int)
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/d))
    k1[k1>=np.sqrt(1/(4*d))] -= np.sqrt(1/d)

    Lu = np.zeros((d,N,samples))
    for i in range(N):
        for j in range(samples):
            Lu[:,i,j] = pu_belief(k1[i,j],sig,u[i,j],d)

    obj.sc_decode(Lu)
    logZ_Wi_list_X = np.max(scipy.special.logsumexp(-obj.LLRs/2,axis=-1),axis=0)

    return (np.exp(logZ_Wi_list_X)).flatten()/(samples)


def disp_sq_dGKP_polar_MC_error(polar_amp, polar_phase, sig, samples):
    """
    polar_bit: cl_polar_dit object with dits frozen in set A
    polar_phase: cl_polar_dit object with dits frozen in set P
    sig: spread of displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    N = polar_amp.N
    d = polar_amp.d
    
    x_error = np.random.normal(0,sig,(N,samples))
    y_error = np.random.normal(0,sig,(N,samples))
    u = np.array(np.rint(x_error/np.sqrt(2*np.pi/d))%d,dtype=int)
    v = np.array(np.rint(y_error/np.sqrt(2*np.pi/d))%d,dtype=int)
        
    k1 = np.mod(x_error/np.sqrt(2*np.pi),np.sqrt(1/d))
    k1[k1>=np.sqrt(1/(4*d))] -= np.sqrt(1/d)
    k2 = np.mod(y_error/np.sqrt(2*np.pi),np.sqrt(1/d))
    k2[k2>=np.sqrt(1/(4*d))] -= np.sqrt(1/d)

    Lu = np.zeros((d,N,samples))
    Lv = np.zeros((d,N,samples))
    for i in range(N):
        for j in range(samples):
            Lu[:,i,j] = pu_belief(-k1[i,j],sig,0,d)
            Lv[:,i,j] = pu_belief(-k2[N-1-i,j],sig,0,d) #Flipped along N
        
    polar_amp.uAc = polar_amp.reverse_encode(u) #this is u'
    polar_phase.uAc = (polar_phase.reverse_encode(np.flip(v,axis=0))) #this is flipped v'

    u_hat = polar_amp.sc_decode(Lu)
    v_hat = np.flip(polar_phase.sc_decode(Lv),axis=0)
    
    avg_u_diff  = np.sum(np.sum((u_hat - u)%d,axis=0))/samples
    avg_v_diff = np.sum(np.sum((v_hat - v)%d,axis=0))/samples
    avg_error_count = np.sum(np.minimum(1,np.sum((u_hat - u)%d,axis=0) + np.sum((v_hat - v)%d,axis=0)))/samples
    
    return avg_error_count, avg_u_diff, avg_v_diff


def no_analog_dGKP_polar_MC_error(polar_amp, polar_phase, sig, samples):
    """
    polar_bit: cl_polar_dit object with dits frozen in set A
    polar_phase: cl_polar_dit object with dits frozen in set P
    sig: spread of displacement noise
    samples: no. of runs for monte-carlo sampling
    """
    N = polar_amp.N
    d = polar_amp.d

    p = np.array([scipy.integrate.quad(pu_integrand,-np.sqrt(np.pi/(2*d)),np.sqrt(np.pi/(2*d)),
                                       args=(sig,i))[0] for i in range(d)])
    L = np.array([np.log(p[0]/p[i]) for i in range(d)]).reshape((d,1,1))
    
    x_error = np.random.normal(0,sig,(N,samples))
    y_error = np.random.normal(0,sig,(N,samples))
    u = np.array(np.rint(x_error/np.sqrt(2*np.pi/d))%d,dtype=int)
    v = np.array(np.rint(y_error/np.sqrt(2*np.pi/d))%d,dtype=int)

    Lu = L*np.ones((d,N,samples))
    Lv = L*np.ones((d,N,samples))
        
    polar_amp.uAc = polar_amp.reverse_encode(u) #this is u'
    polar_phase.uAc = (polar_phase.reverse_encode(np.flip(v,axis=0))) #this is flipped v'

    u_hat = polar_amp.sc_decode(Lu)
    v_hat = np.flip(polar_phase.sc_decode(Lv),axis=0)
    
    avg_u_diff  = np.sum(np.sum((u_hat - u)%d,axis=0))/samples
    avg_v_diff = np.sum(np.sum((v_hat - v)%d,axis=0))/samples
    avg_error_count = np.sum(np.minimum(1,np.sum((u_hat - u)%d,axis=0) + np.sum((v_hat - v)%d,axis=0)))/samples
    
    return avg_error_count, avg_u_diff, avg_v_diff