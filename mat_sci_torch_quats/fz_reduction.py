from quat_conversion import cubo2quat, quat2cubo, quat2rod, rod2quat, rod2cubo
from quats import Quat
from symmetries import hcp_syms, fcc_syms
import numpy as np
from time import time


# Generate random cubochoric coordinates

N = 10000000
symmetry_system = 'hcp'

rand_cubos = np.random.uniform(low = -1*np.pi/2, high = np.pi/2, size = (N, 3))

t1 = time()

# Convert to quaternions and assign as Quats
conv_quats = Quat(cubo2quat(rand_cubos, scalar_first=False))

# Preallocate ndarrays for symmetry and perform outer product
quat_syms = 0
symcount = 0
if symmetry_system is 'hcp':
    quat_syms = conv_quats.outer_prod(hcp_syms)
    symcount = 12
else:
    quat_syms = conv_quats.outer_prod(fcc_syms)
    symcount = 24

#####################
# Loss calcualtiona with expanded quaternion symmetry would be calculated here using quat_syms
####################

# Create nadarray for Rodrigues expansion
rod_syms = np.zeros((N, symcount, 4))

# Perform Rodrigues conversion on expanded symmetries
for current_sym in range(0, symcount):
    rod_syms[:, current_sym, :] = quat2rod(quat_syms.X[:, current_sym])


# For each symmetry expanded Quat, find index for lowest magnitude value of tan(omega/2)
rod_filter = np.argmin((np.abs(rod_syms)), axis=1)[:, -1]

# rod_filter is a 1d array of length N containing the index of the minimum magnitude
# Rodrigues vector for each set of symmetry expanded Quats.  This index corresponds to the
# second index value in rod_syms.  For example, if 2 HCP Quats are being symmetry expanded in
# Rodrigues space, rod_syms will be of size:
# [2(number of Quats), 12(number of symmetries), 4(number of Rodrigues terms)]
# and rod_filter will be of size [2(number of Quats)], and will contain the indices corresponding
# to which symmetry index (0-11) has the lowest magnitude value of tan(omega/2)

# Based on these indices, extract the min magnitude symmetry form for each quat
rod_reduced = rod_syms[np.arange(0,N), rod_filter,:]

# Convert back to cubochoric space
cubo_reduced = rod2cubo(rod_reduced)

print("Conversion time for "+str(N)+" Samples = "+str(time()-t1)+" seconds")

