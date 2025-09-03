"""

Monte Carlo simulation methods for the Voter Model


We inherit the main class for it


"""

import numpy as np
import votermain as vm
from votermain import OneDVoter
from numpy import random as rnd



class VoterSim(OneDVoter):
    """
    
    The class for simulation methods mainly focused on
    
    the continious time markov chain simulations with
    
    the Gillespie algortihm, the class inherits the 
    
    OneDVoter and initializes with parameters
    
    N: system size
    alpha: scaling factor
    
    """
    def random_lattice(self):
        """
        Get an N sized random lattice
        by construct it doesn't allow the consuming states like [1,1,1]

        Returns
        -------
        latt : random initial lattice

        """
    
        latt = np.zeros(self.N)
        
        for i in range(self.N):
            
            u = rnd.choice([1,0],1)
            latt[i] = u
            
        if np.sum(latt==1)==self.N:
            a = rnd.choice(np.arange(0,self.N),1)
            latt[a] = 0
        elif np.sum(latt==0)==self.N:
            a = rnd.choice(np.arange(0,self.N),1)
            latt[a] = 1
            
        return latt
    
    def gil_single_step(self,latt):
        """
        Single Gillespie step
        
        The idea is to calculate only the possible transitions from the 
        
        given latt and use those rates to deduce which one we obatin (i)
        
        within the time frame tau

        Parameters
        ----------
        latt : Current state of the system

        Returns
        -------
        trates : Transition rates of possible states
        tau : The time it takes to transiton
        i : Next state
        """

        
        state_num = vm.bin_to_number(latt)

        
        trates = self.trans_rates(state_num)
        
        t_keys = list(trates.keys())
        
        l_c = sum(list((trates.values())))
        
        rnd1 = rnd.uniform(0,1)
        
        tau = -np.log(1-rnd1)/l_c
        
        rnd2 = rnd.uniform(0,1)*l_c
        
        i = -1
        r = 0
        
        while r<rnd2:
            
            i+=1
            r+= trates[t_keys[i]]
            
        
        return trates,tau,i
    
    def get_magne(self,latt):
        """
        Get the per spin magnetizaiton from the current state 

        Parameters
        ----------
        latt : Current state of the system

        Returns
        -------
        Per spin magnetization

        """
        lat2 = np.copy(latt)
        
        lat2[lat2==0]=-1
        
        return np.sum(lat2)/len(lat2)
    
    def time_sim(self,is_random,tmax):
        """
        Full time simulation for the system
        mainly operates with the gil_single_step()
        We obtain the most important data from the realized trajectories
        

        Parameters
        ----------
        is_random : If True we start with a random initial lattice otherwise
        it takes a user input in the form of 1,0,0,1,... etc
        tmax : Maximum time allocated for the sim

        Returns
        -------
        tvals : Overall time it takes for the sim
        tvals2 : The time steps between transitions
        magnes : The change of magnetizaiton over time (tvals)
        epr : The entropy produciton rate for each transition the 

        """
    
        tvals = [0]
        tvals2 = []
        epr = []
        magnes = []
        t = 0
        
        if is_random:
        
            init_lat = self.random_lattice()

            
        else:
            
            user_state = list(input("Enter the state values: ").split(","))
            init_lat = np.array(user_state,dtype=int)
        
        while t<tmax:

            magne = self.get_magne(init_lat)
            magnes.append(magne)
            
            
            trans, tau, i = self.gil_single_step(init_lat)
            t+=tau
            tvals.append(t)
            tvals2.append(tau)
            
            t_keys = list(trans.keys())
            
            trate1 = vm.get_mat_el(t_keys[i])
            inv_tr = str(trate1[1])+":"+str(trate1[0])
            
            inv_rates = self.trans_rates(trate1[1])
            
            ent1 = np.log(trans[t_keys[i]]/inv_rates[inv_tr])   
    
            epr.append(ent1)
            
            init_lat = vm.number_to_bin(trate1[1],self.N)
        
        magnes.append(self.get_magne(init_lat))
        
        return tvals,tvals2, magnes,epr
        
        