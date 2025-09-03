"""

The main class for the one dimensional voter model with long range interactions

The instance is defined by two main parameters N and alpha.

N is the size of the one dimensional regular spin chain.

alpha is for the range of interaction alpha=0 infinite range for all 

interacting spins and alpha=inf basically nearest neighbor interactions.



"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs


def number_to_bin(x,N):
    """
    Takes the decimal value of a state with size N and
    
    returns the binary representation of spins 
    
    Give the deprecated error but functions well
    
    Should be updated!!

    Parameters
    ----------
    x : Numerical value for the given bit
    N : Total number of bits

    Returns
    -------
    main_array : Array of bit to string representation

    """
    main_array = np.fromstring(np.binary_repr(x), dtype='S1').astype(int)
    if len(main_array)<N:
        main_array = np.append(np.zeros(N-len(main_array)),main_array)
    return main_array


def bin_to_number(vec):
    """
    Gets the decimal representation of a given bit array (spin state)

    Parameters
    ----------
    vec : Given bit array

    Returns
    -------
    u : Number

    """
    u = 0
    for i in range(len(vec)):
        if vec[len(vec)-1-i]==1:
            u+=2**i
    return u

def get_mat_el(wij_key):
    """
    Takes the keys form the transition rate dictionary
    
    and returns matrix elements in tuple form
    
    

    Parameters
    ----------
    wij_key : Key from transition rate array

    Returns
    -------
    Matrix element tuple

    """
    i = 0
    j = ""
    k = ""
    while wij_key[i] != ":":
        j += wij_key[i]
        i+=1
    i+=1
    while i<len(wij_key):
        k+= wij_key[i]
        i+=1
    return (int(j),int(k))


class OneDVoter:
    """
    Main class to define 1D Voter Model calculations
    
    Initializaiton only needs N and alpha 
    
    """
    
    def __init__(self,N,alpha):
        """

        Parameters
        ----------
        N : The system size
        alpha : Scaling factor for interaction range
        Returns
        -------
        None.

        """
        self.N = N
        self.alpha = alpha
    
    def get_Z(self):
        """
        Function to obtain normalization constant for the trans rates
        Works weirdly but works
    
        Parameters
        ----------
        N : Number of spins
        alpha : Range parameter
    
        Returns
        -------
        Normalization constant
        """
        Z=0
        list1 = [i for i in range(1,self.N)]
        list3 = []
        for i in range(0,int(self.N/2)):
            list2= list1[0:i]
            if i==0:
                list3=list1
            else:   
                
                for j in range(i):
                    list4 = [list2[j],list2[j]]
                    list3 +=list4
                for k in range(2*i,self.N-1):
                    list3.append(k)
            for l in list3:
                Z+=l**(-self.alpha)
            list3=[]
        return 2*Z
    
    
    def trans_rates(self,i):
        """
        Calculate transition rates from a given state in decimal
        
        for example i=1 for a system size 4 is the state [0,0,0,1]
        
        then we get every possible transition from this state
        
        the transitions to states [0...0...0] or [1...1...1] where no back
        
        transition is possible are 0 by construction
        
    
        Parameters
        ----------
        i : Given state

        Returns
        -------
        wij : Dictionary for the every transition from given i to 
        possible j's
    
        """
        Z = self.get_Z()
        
        state_arr = number_to_bin(i,self.N)
        wij = {}
        trate = 0
        for j in range(self.N-1,-1,-1):
            if state_arr[j]==0:
                new_state = np.copy(state_arr)
                new_state[j]=1
            elif state_arr[j]==1:
                new_state = np.copy(state_arr)
                new_state[j]=0
            if bin_to_number(new_state)==0 or \
            bin_to_number(new_state)==2**self.N-1:
                continue
            for b in range(self.N):
                if state_arr[j]==state_arr[b]:
                    continue
                elif state_arr[j]!=state_arr[b]:
                    trate+= (abs(j-b)**(-self.alpha))/Z
            # This is an arbitrary choice on having a lower bound for 
            # smallest transition rates if no lower bound is introduced
            # very small rates are considered and unrealistic transitions 
            # are taken into account (blows up epr calculations)
            if trate>10E-20:
                wij[str(bin_to_number(state_arr))+":"+ 
                    str(bin_to_number(new_state))]=trate
            
            trate = 0
        return wij
    def full_rates(self):
        """
        Get all transition rates as a dictionary for our predefined system
        
        in addition we calculate the diagonal elements of the transition 
        
        matrix.
    
    
        Returns
        -------
        w1j : All transition rates
        diag_ij: Diagonal elements of the transition matrix
    
        """
        
        w1j = self.trans_rates(1)
        diag_ij = {}
        diag_ij["1:1"] = -sum(list((w1j.values())))
        for i in range(2,2**self.N-1):
            
            wij = self.trans_rates(i)
            aij = -sum(list((wij.values())))
            diag_ij[str(i)+":"+ 
                str(i)]=aij
            w1j.update(wij)
            
        return w1j,diag_ij
    
    def get_matrix(self):
        """
        The method to get the full transition matrix for initialized system
        
        Should be noted that the matrix size is (2**N,2**N)
        
        therefore, for N>20 there will be memory problems to store and 
        
        do calculations over it.
    
       
        Returns
        -------
        Aij : Transiiton matrix
    
        """
        
        
        Aij = np.zeros((2**self.N,2**self.N))
        
        i = 1
        
        while i<2**self.N:
            wij = self.trans_rates(i)
            l_i = sum(list((wij.values())))
            Aij[i,i] = -l_i
            for n in range(len(wij)):
                keys1 = list(wij.keys())
                el = get_mat_el(keys1[n])
                Aij[el]=wij[keys1[n]]
            i+=1
            
        return Aij
    
    def exact_EPR(self):
        """
        This method calculates the exact Entropy Production Rate for our given
        
        system. Extremely slow as it calculates eigenvalues/vectors from the 
        
        transition matrix (2**N,2**N)
    
    
        Returns
        -------
        ent_r : Entropy produciton rate
    
        """
        Aij = self.get_matrix()
        
        u,w = np.linalg.eigh(Aij.T)
        
        Pi = abs(w[:,np.argmax(u)])/np.sum(abs(w[:,np.argmax(u)]))
        
        ent_r = 0
        
        for i in range(2**self.N):
            for j in range(2**self.N):
                if i==j:
                    continue
                elif Aij[i,j]==0 or Aij[j,i]==0:
                    continue
                ent_r += Pi[j]*Aij[i,j]*\
                np.log((Pi[i]*Aij[i,j])/(Pi[j]*Aij[j,i]))
                
        return ent_r
        
    def sparse_eigv(self):
        """
        To approxiamte steady states for larger systems (N>20) we use
        
        the sparse matrix methods. This method calculates the steady states
        
        for our given system
    
    
        Returns
        -------
        Pi : Steady state prob distribution of the system
    
        """
        
        wij,diag_ij = self.full_rates()
        
        def matmul1(v):
            """
            Matrix multoplication module to use as a LinearOperator
            for the scipy sparse eigenvalue solver
    
    
            Parameters
            ----------
            v : Input vector with dim 2**N
    
            Returns
            -------
            v2 : Output vector multiplied with the transition matrix
    
            """
            v2 = np.zeros(len(v))
            keys1 = list(wij.keys())
            keys2 = list(diag_ij.keys())
            for i in range(len(wij)):
                el = get_mat_el(keys1[i])
                v2[el[1]]+= v[el[0]]*wij[keys1[i]]
            for j in range(len(diag_ij)):
                el = get_mat_el(keys2[j])
                v2[el[1]]+= v[el[0]]*diag_ij[str(el[0])+":"+str(el[0])]
            return v2
        
        A = LinearOperator((2**self.N,2**self.N),matvec=matmul1)
        
        vals, vecs = eigs(A,k=1)
        
        Pi = abs(vecs[:,0])/np.sum(abs(vecs[:,0]))
        
        return Pi
    
    def sparse_entro(self):
        """
        To get approximate results for large system entropy production rate
        
        we use this module in conjunction with the sparse_eigv.
        
        Note that the bottleneck of this method is the call of full_rates()
        
        and the sparse_eigv() in some cases it could be better to save results
        
        from these functions and use this module in a seperate script as a
        
        function.
    
    
        Returns
        -------
        ent_r : Entropy production rate
    
        """
        ent_r = 0
        w,dij = self.full_rates()
        v = self.sparse_eigv()
        keys1 = list(w.keys())
        
        for i in range(len(keys1)):
            el = get_mat_el(keys1[i])
            P_i = v[el[0]]
            P_j = v[el[1]]
            wij = w[keys1[i]]
            new_key = str(el[1])+":"+str(el[0])
            try: 
                wji = w[new_key]
            except KeyError:
                continue
            ent_r+= (P_i*wij)*np.log((P_i*wij)/(P_j*wji))
        return ent_r
    
        
        