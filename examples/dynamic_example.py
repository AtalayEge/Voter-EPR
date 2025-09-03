"""

Example of simulating the Voter model with the Gillespie algorithm

we will focus on obtaining EPR from dynamic calulations

"""


from votersim import VoterSim
import numpy as np 
import matplotlib.pyplot as plt



# Set up the system sizes

N1 = 8

N2 = 16

# Set up alpha values

alpha = [0,1,2,3,4,5]

# max time

tmax = 10**3

# Epr vals for each alpha

epr_valsN1 = np.zeros(len(alpha))
epr_valsN2 = np.zeros(len(alpha))

# Set the simulation time

for i in range(len(alpha)):
    
    # Set the instance
    
    u1 = VoterSim(N1, alpha[i])
    u2 = VoterSim(N2, alpha[i])
    
    t1,t2,magnes,epr = u1.time_sim(True, tmax)
    t3,t4,magnes,epr2 = u2.time_sim(True, tmax)
    
    # sum each epr contirubution and divide by the total time of sim
    
    epr_valsN1[i] = np.sum(epr)/t1[-1]
    epr_valsN2[i] = np.sum(epr2)/t3[-1]
    
    
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\sigma$")

plt.plot(alpha,epr_valsN1,"-ok")
plt.plot(alpha,epr_valsN2,"-or")
plt.legend(["Dynamic EPR, N=8", "Dynamic EPR, N=16"])

"""
We get quite different results each time as calculated EPR depends

heavily on the trajectory, to get more reliable results epr should be calculated

for longer tmax especially for larger systems and averaged out for many

simulations with errors calculated

"""

    