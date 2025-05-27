import numpy as np # type: ignore
import scipy # type: ignore

#Error analysis for simple displacement channel
def pu(k1, sig, u=1, d=2, l_cutoff=8):
    all_index = np.array(range(-l_cutoff*d, l_cutoff*d + 1))
    u_list = np.array(range(-l_cutoff*d + u, l_cutoff*d + 1, d))
    log_numer = scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_list*np.sqrt(2*np.pi/d))**2)/(2*sig*sig))
    log_denom = scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - all_index*np.sqrt(2*np.pi/d))**2)/(2*sig*sig))
    return np.exp(log_numer-log_denom)
pu = np.vectorize(pu)

def pu_belief(k1,sig,u,d,l_cutoff=8):
    u_lists = np.array([range(-l_cutoff*d + i, l_cutoff*d, d) for i in range(d)])
    logpu = scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_lists*np.sqrt(2*np.pi/d))**2)/(2*sig*sig),axis=-1)
    return np.array([logpu[-int(u)] - logpu[(-int(u)+i)%d] for i in range(d)])
pu_belief = np.vectorize(pu_belief)

def pu_integrand(x,sig,u=1,d=2,l_cutoff=10):
    u_list = np.array(range(-l_cutoff*d + u, l_cutoff*d + 1, d))
    log_numer = scipy.special.logsumexp(-((x - u_list*np.sqrt(2*np.pi/d))**2)/(2*sig*sig))
    return np.exp(log_numer)/(np.sqrt(2*np.pi)*sig)
            
def Hp_integrand(k1,sig,d=2,l_cutoff=10):
    all_index = np.array(range(-l_cutoff*d, l_cutoff*d))
    u_lists = np.array([range(-l_cutoff*d + u, l_cutoff*d, d) for u in range(d)])
    p = -np.log(sig)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - all_index*np.sqrt(2*np.pi/d))**2)/(2*sig*sig))
    pu = -np.log(sig)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_lists*np.sqrt(2*np.pi/d))**2)/(2*sig*sig),axis=1)
    return -(np.sum(pu*np.exp(pu)) - p*np.exp(p))/np.log(2)

def Hp_rect_integrand(k1,factor,sig,d=2,l_cutoff=10):
    all_index = np.array(range(-l_cutoff*d, l_cutoff*d))
    u_lists = np.array([range(-l_cutoff*d + u, l_cutoff*d, d) for u in range(d)])
    sigb = sig*factor
    sigp = sig/factor
    p_bitf = -np.log(sigb)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - all_index*np.sqrt(2*np.pi/d))**2)/(2*sigb*sigb))
    pu_bitf = -np.log(sigb)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_lists*np.sqrt(2*np.pi/d))**2)/(2*sigb*sigb),axis=1)
    p_phsf = -np.log(sigp)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - all_index*np.sqrt(2*np.pi/(d)))**2)/(2*sigp*sigp))
    pu_phsf = -np.log(sigp)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_lists*np.sqrt(2*np.pi/(d)))**2)/(2*sigp*sigp),axis=1)
    return (-(np.sum(pu_bitf*np.exp(pu_bitf)) - p_bitf*np.exp(p_bitf))-(np.sum(pu_phsf*np.exp(pu_phsf)) - p_phsf*np.exp(p_phsf)))/np.log(2)



#Finite energy 
def p1_finite(k1, sig_gkp, sig, l_cutoff=20):

    integrand = lambda k_anc: np.exp(-k_anc**2/(2*sig_gkp*sig_gkp/(2*np.pi)))*pu(k1+k_anc,sig,l_cutoff=l_cutoff)/sig_gkp
    return scipy.integrate.quad(integrand,-3*sig_gkp,3*sig_gkp)[0]

def finite_Hp_integrand(k1, sig_gkp, sig, l_cutoff=20):

    def integrand(k_anc):
        p0 = 0.0
        p1 = 0.0
        p = 0.0
        for i in range(-l_cutoff,l_cutoff+1,1):
            x = np.exp(-(((k1+k_anc)*np.sqrt(2*np.pi) - i*np.sqrt(np.pi))**2)/(2*sig*sig))/(sig)
            p += x
            p1 += x*float(i%2 != 0)
            p0 += x*float(i%2 == 0)
        return np.exp(-k_anc**2/(2*sig_gkp*sig_gkp/(2*np.pi)))*np.array([p0,p1,p])/sig_gkp
    
    p0,p1,p = scipy.integrate.quad_vec(integrand,-3*sig_gkp,3*sig_gkp)[0]
    return (scipy.special.entr(p0) + scipy.special.entr(p1) - scipy.special.entr(p))/np.log(2)


#Hexagonal GKP
hex_expr = lambda s1,s2,b1,b2,d: 2*(b1*b1 + b2*b2 + b2*s1 + s1*s1 - 
                                    (2*b2+s1)*s2 + s2*s2 + b1*(-b2-2*s1+s2))/(np.sqrt(3)*d) 
def b0_hex(s1,s2):
    all_expr = np.array([[hex_expr(s1,s2,i,j,2) for i in [-1,0,1]] for j in [-1,0,1]])
    index = np.argmin(all_expr.flatten())
    b1 = index%3-1
    b2 = index//3 - 1
    return b1,b2
b0_hex = np.vectorize(b0_hex)

def p_uv_hex(k,kk,u,v,sig,d,l_cutoff=4):
    logp_uv = -np.inf
    b1,b2 = b0_hex(k,kk)
    k_new = k - b1
    kk_new = kk - b2
    for l in range(-l_cutoff,l_cutoff+1,1):
        for ll in range(-l_cutoff,l_cutoff+1,1):        
            x = (k_new-2*kk_new + 2*d*l - 2*v + d*ll - u)/np.sqrt(np.sqrt(3)*2*d)
            y = (k_new + d*ll - u)*np.sqrt(np.sqrt(3)/(2*d))
            logp_uv = np.logaddexp(logp_uv,np.log(1/(2*sig*sig)) - np.pi*(x**2+y**2)/(sig*sig))
    return np.exp(logp_uv)

def Hp_hex_integrand(s1,s2,sig,d,l_cutoff=4):
    logp_uv = -np.inf + np.zeros((d,d))
    log_ptot = -np.inf
    b1,b2 = b0_hex(s1,s2)
    k_new = s1 - b1
    kk_new = s2 - b2
    for l in range(-d*l_cutoff,d*l_cutoff+1,1):
        for ll in range(-d*l_cutoff,d*l_cutoff+1,1):        
            x = (k_new - 2*kk_new + 2*l + ll)/np.sqrt(np.sqrt(3)*2*d)
            y = (k_new + ll)*np.sqrt(np.sqrt(3)/(2*d))
            logp_uv[l%d,ll%d] = np.logaddexp(logp_uv[l%d,ll%d],np.log(1/(2*sig*sig)) - np.pi*(x**2+y**2)/(sig*sig))
            log_ptot = np.logaddexp(log_ptot,np.log(1/(2*sig*sig)) - np.pi*(x**2+y**2)/(sig*sig))
    return (np.sum(np.sum(logp_uv*np.exp(logp_uv))) - np.exp(log_ptot)*(log_ptot))/np.log(2)

#Rectangular GKP
def Hp_rect_integrand(k1,factor,sig,d=2,l_cutoff=10):
    all_index = np.array(range(-l_cutoff*d, l_cutoff*d))
    u_lists = np.array([range(-l_cutoff*d + u, l_cutoff*d, d) for u in range(d)])
    sigb = sig*factor
    sigp = sig/factor
    p_bitf = -np.log(sigb)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - all_index*np.sqrt(2*np.pi/d))**2)/(2*sigb*sigb))
    pu_bitf = -np.log(sigb)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_lists*np.sqrt(2*np.pi/d))**2)/(2*sigb*sigb),axis=1)
    p_phsf = -np.log(sigp)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - all_index*np.sqrt(2*np.pi/(d)))**2)/(2*sigp*sigp))
    pu_phsf = -np.log(sigp)+scipy.special.logsumexp(-((k1*np.sqrt(2*np.pi) - u_lists*np.sqrt(2*np.pi/(d)))**2)/(2*sigp*sigp),axis=1)
    return (-(np.sum(pu_bitf*np.exp(pu_bitf)) - p_bitf*np.exp(p_bitf))-(np.sum(pu_phsf*np.exp(pu_phsf)) - p_phsf*np.exp(p_phsf)))/np.log(2)