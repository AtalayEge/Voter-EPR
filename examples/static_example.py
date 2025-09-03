"""

Example of calculating entropy production rate for the Voter Model

from the full transition rates

"""

import numpy as np
import matplotlib.pyplot as plt
from votermain import OneDVoter



# Set up the system size

N = 8

# Set up alpha values

alpha = [0,1,2,3,4,5]

# Epr vals for each alpha (exact and approximate)

exact_epr_vals = np.zeros(len(alpha))
approx_erp_vals = np.zeros(len(alpha))

# Loop over for each alpha val

for i in range(len(alpha)):
    
    # Get the instance
    
    u = OneDVoter(N, alpha[i])
    
    exact_epr_vals[i] = u.exact_EPR()
    approx_erp_vals[i] = u.sparse_entro()
    

# Plot the obtained values alpha vs epr

plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\sigma$")

plt.plot(alpha,exact_epr_vals,"-ok")
plt.plot(alpha,approx_erp_vals,"-or")
plt.legend(["Exact EPR, N=8", "Approx EPR, N=8"])
    

    
    


