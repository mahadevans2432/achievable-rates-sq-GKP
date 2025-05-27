import numpy as np # type: ignore
import scipy # type: ignore

f = np.vectorize(lambda a,b: np.logaddexp(0,-a-b) - np.logaddexp(-a,-b))
g = np.vectorize(lambda a,b,c: b + (np.power(-1,c))*a)
bit = np.vectorize(lambda L: (int(L<0)))

def f_dit(L1,L2):
    """
    L1 & L2 have shape[0] = d
    """
    d = L1.shape[0]
    num = scipy.special.logsumexp(-(L1 + L2),axis=0,keepdims=True)
    denoms = np.array([scipy.special.logsumexp(-(np.roll(L1,-i,axis=0) + L2),axis=0) for i in range(d)])
    return num - denoms
def g_dit(L1,L2,u):
    d,N,samples = L1.shape
    u = np.astype(u,int)
    k_vals = np.arange(d).reshape(d, 1, 1)  # Shape (d, 1, 1) for broadcasting
    u_shifted = (u[np.newaxis, :, :] + k_vals) % d  # Shape (d, N, samples)
    Lg = L2 - L1[u[np.newaxis, :, :], np.arange(N)[:, None], np.arange(samples)] + L1[u_shifted, np.arange(N)[:, None], np.arange(samples)]
    return Lg

def f_dit_alpha(L1,L2,alpha):
    """
    L1 & L2 have shape[0] = d
    """
    d = L1.shape[0]
    L2_permuted = L2*0
    for j in range(d):
        L2_permuted[(alpha*j)%d,:,:] = L2[j,:,:]
    num = scipy.special.logsumexp(-(L1 + L2_permuted),axis=0,keepdims=True)
    denoms = np.array([scipy.special.logsumexp(-(np.roll(L1,-i,axis=0) + L2_permuted),axis=0) for i in range(d)])
    return num - denoms
def g_dit_alpha(L1,L2,u,alpha):
    d,N,samples = L1.shape
    u = np.astype(u,int)
    k_vals = np.array([(alpha*i)%d for i in range(d)]).reshape(d, 1, 1)  # Shape (d, 1, 1) for broadcasting
    u_shifted = (u[np.newaxis, :, :] + k_vals) % d  # Shape (d, N, samples)
    Lg = L2 - L1[u[np.newaxis, :, :], np.arange(N)[:, None], np.arange(samples)] + L1[u_shifted, np.arange(N)[:, None], np.arange(samples)]
    return Lg

dit = lambda L: (np.argmin(L[1:],axis=0) + 1)*bit(np.amin(L[1:],axis=0))

class cl_polar:

    def __init__(self, N, A, uAc = None):
        """
        N: the number of bits of the code
        A: list of indices (zero indexed) which have information encoded
        uAc: list with frozen indices storing the frozen bit value and the rest of indices have None assigned
        """
        self.N = N
        self.K = len(A)
        self.A = A
        if uAc == None:
            self.uAc = np.array([0]*N)
            for i in range(N):
                if not(i in A):
                    self.uAc[i] = 0
        else:
            self.uAc = uAc

    def _pass_up(self,left,right):
        return np.append(np.array([(left[i]+right[i])%2 for i in range(len(left))]),right)
    
    def _pass_bit_down_left(self,x):
        if len(x)==1:
            return x
        return (x[:len(x)//2] + x[len(x)//2:])%2

    def _pass_bit_down_right(self,x):
        if len(x)==1:
            return x
        return x[len(x)//2:]
    
    def _pass_bit_down_right_decoding(self,x,u):
        if len(x)==1:
            return x
        return (x[:len(x)//2] + u)%2
    
    def _pass_down_left(self,x):
        if len(x) == 1:
            return 0
        return f(x[:len(x)//2],x[len(x)//2:])
    
    def _pass_down_right(self, x, u):
        if len(x) == 1:
            return 0
        return g(x[:len(x)//2],x[len(x)//2:],u)
    

    def freeze_and_encode(self,u):
        """
        u: bitstring of length K = |A|
        This function freezes the remaining N-K bits and sends to _encode
        """
        ue = []
        j=0
        for i in range(self.N):
            if i in self.A:
                ue.append(u[j])
                j += 1
            else:
                ue.append(self.uAc[i])
        return self._encode(ue)

    def _encode(self, u):
        """
        u: bitstring of length N with frozen bits in Ac set to right values
        """
        if len(u)==1:
            return u
        return self._pass_up(self._encode(u[:len(u)//2]),self._encode(u[len(u)//2:]))

    
    def bpsk_awgn(self,bits,sig):
        return [(1-2*b) + np.random.normal(0,sig) for b in bits]
    

    def sc_decode(self, L, indices=None):
        """
        L: list of beliefs
        returns: decoded guess based on SC decoder
        """
        if len(L) == self.N:
            self.LLRs = np.array([0.0 for i in range(self.N)])
            indices = [i for i in range(self.N)]
        
        if len(L) == 1:
            i = indices[0]
            self.LLRs[i] = L[0]
            if not (i in self.A):
                return [self.uAc[i]]
            else:
                return [bit(L[0])]

        left_L = f(L[:len(L)//2],L[len(L)//2:])
        u1 = self.sc_decode(left_L,indices[:len(L)//2])
        right_L = g(L[:len(L)//2],L[len(L)//2:],u1)
        u2 = self.sc_decode(right_L,indices[len(L)//2:])
        
        return self._pass_up(u1,u2)
    
    def hard_sc_decode(self, bits, delta):
        """
        Decodes N bits
        Assumes noise channel of delta probability bit-flip
        """
        c = np.log(1-delta) - np.log(delta)
        return self.sc_decode([c*(1-2*b) for b in bits])


    def bit_decode(self, bits, indices=None):
        bits = np.array(bits)
        if len(bits) == self.N:
            indices = [i for i in range(self.N)]
        
        if len(bits) == 1:
            i = indices[0]
            if not (i in self.A):
                return [self.uAc[i]]
            else:
                return [bits[0]]

        u1 = self.bit_decode(self._pass_bit_down_left(bits),indices[:len(bits)//2])
        u2 = self.bit_decode(self._pass_bit_down_right_decoding(bits,u1),indices[len(bits)//2:])
        return self._pass_up(u1,u2)
    

class cl_polar_parallel:

    def __init__(self, N, A, samples, uAc = None):
        """
        Parallelizable version of cl_polar
        N: the number of bits of the code
        A: list of indices (zero indexed) which have information encoded
        uAc: list with frozen indices storing the frozen bit value and the rest of indices have None assigned
        """
        self.N = N
        self.K = len(A)
        self.A = A
        self.samples = samples
        if uAc == None:
            self.uAc = np.zeros((N,samples))
        else:
            self.uAc = uAc

    def _pass_up(self,left,right):
        return np.append((left+right)%2,right,axis=0)

    def _encode(self, u):
        """
        u: bitstring of length N with frozen bits in Ac set to right values
        """
        if len(u)==1:
            return u
        return self._pass_up(self._encode(u[:len(u)//2]),self._encode(u[len(u)//2:]))
    

    def sc_decode(self, L, indices=None):
        """
        L: list of beliefs
        returns: decoded guess based on SC decoder
        """
        if L.shape[0] == self.N:
            self.LLRs = 0*L
            indices = [i for i in range(self.N)]
        
        if L.shape[0] == 1:
            i = indices[0]
            self.LLRs[i,:] = L[0,:]
            if not (i in self.A):
                return self.uAc[i,:].reshape((1,self.samples))
            else:
                return bit(L[0,:]).reshape((1,self.samples))

        left_L = f(L[:len(L)//2],L[len(L)//2:])
        u1 = self.sc_decode(left_L,indices[:len(L)//2])
        right_L = g(L[:len(L)//2],L[len(L)//2:],u1)
        u2 = self.sc_decode(right_L,indices[len(L)//2:])
        
        return self._pass_up(u1,u2)


class cl_polar_dit:

    def __init__(self, d, N, A, samples, uAc = None):
        """
        Class for dit based classical polar codes
        d: dit level
        N: the number of bits of the code
        A: list of indices (zero indexed) which have information encoded
        samples: no of samples to parallelize operations over
        uAc: list with frozen indices storing the frozen bit value and the rest of indices have None assigned
        """
        self.d = d
        self.N = N
        self.K = len(A)
        self.A = A
        self.samples = samples
        if uAc == None:
            self.uAc = np.zeros((N,samples))
        else:
            self.uAc = uAc

    def _pass_up(self,left,right):
        return np.append((left+right)%self.d,right,axis=0)
    
    def _rev_pass_up(self,left,right):
        return np.append((left-right)%self.d,right,axis=0)

    def _encode(self, u):
        """
        u: bitstring of length N with frozen bits in Ac set to right values
        """
        if len(u)==1:
            return u
        return self._pass_up(self._encode(u[:len(u)//2]),self._encode(u[len(u)//2:]))
    
    def reverse_encode(self, u):
        """
        Applies G_N^{-1}
        """
        if len(u)==1:
            return u
        return self._rev_pass_up(self.reverse_encode(u[:len(u)//2]),self.reverse_encode(u[len(u)//2:]))

    def sc_decode(self, L, indices=None):
        """
        L: list of beliefs in shape (d, N, samples)
        returns: decoded guess based on SC decoder
        """
        if L.shape[1] == self.N:
            self.LLRs = np.zeros((self.d-1,self.N,self.samples))
            indices = [i for i in range(self.N)]
        
        if L.shape[1] == 1:
            i = indices[0]
            self.LLRs[:,i,:] = L[1:,0,:]
            if not (i in self.A):
                return self.uAc[i,:].reshape((1,self.samples))
            else:
                return dit(L[:,0,:]).reshape((1,self.samples))
            
        N = L.shape[1]
        left_L = f_dit(L[:,:N//2,:],L[:,N//2:,:])
        u1 = self.sc_decode(left_L,indices[:N//2])
        right_L = g_dit(L[:,:N//2,:],L[:,N//2:,:],u1)
        u2 = self.sc_decode(right_L,indices[N//2:])
        
        return self._pass_up(u1,u2)
    

class cl_polar_dit_alpha:

    def __init__(self, d, N, A, samples, alpha = 1, uAc = None):
        """
        Class for dit based classical polar codes with inverted kernel
        d: dit level
        N: the number of bits of the code
        A: list of indices (zero indexed) which have information encoded
        samples: no of samples to parallelize operations over
        uAc: list with frozen indices storing the frozen bit value and the rest of indices have None assigned
        """
        self.d = d
        self.N = N
        self.K = len(A)
        self.A = A
        self.samples = samples
        self.alpha = alpha
        if uAc == None:
            self.uAc = np.zeros((N,samples))
        else:
            self.uAc = uAc

    def _pass_up(self,left,right):
        return np.append((left + self.alpha*right)%self.d,right,axis=0)
    
    def _rev_pass_up(self,left,right):
        return np.append((left - self.alpha*right)%self.d,right,axis=0)

    def _encode(self, u):
        """
        u: bitstring of length N with frozen bits in Ac set to right values
        """
        if len(u)==1:
            return u
        return self._pass_up(self._encode(u[:len(u)//2]),self._encode(u[len(u)//2:]))
    
    def reverse_encode(self, u):
        """
        Applies G_N^{-1}
        """
        if len(u)==1:
            return u
        return self._rev_pass_up(self.reverse_encode(u[:len(u)//2]),self.reverse_encode(u[len(u)//2:]))

    def sc_decode(self, L, indices=None):
        """
        L: list of beliefs in shape (d, N, samples)
        returns: decoded guess based on SC decoder
        """
        if L.shape[1] == self.N:
            self.LLRs = np.zeros((self.d-1,self.N,self.samples))
            indices = [i for i in range(self.N)]
        
        if L.shape[1] == 1:
            i = indices[0]
            self.LLRs[:,i,:] = L[1:,0,:]
            if not (i in self.A):
                return self.uAc[i,:].reshape((1,self.samples))
            else:
                return dit(L[:,0,:]).reshape((1,self.samples))
            
        N = L.shape[1]
        left_L = f_dit_alpha(L[:,:N//2,:],L[:,N//2:,:],self.alpha)
        u1 = self.sc_decode(left_L,indices[:N//2])
        right_L = g_dit_alpha(L[:,:N//2,:],L[:,N//2:,:],u1,self.alpha)
        u2 = self.sc_decode(right_L,indices[N//2:])
        
        return self._pass_up(u1,u2)