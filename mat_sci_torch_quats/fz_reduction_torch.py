import quat_conversion_torch
from quats import Quat
from symmetries import hcp_syms, fcc_syms
from time import time
import torch
import math


# Generate random cubochoric coordinates

torchpi = torch.tensor(math.pi)
N = 10
symmetry_system = 'hcp'

rand_cubos = (torchpi/2 - (-1*torchpi/2)) * torch.rand(N, 3) + (-1*torchpi/2)

t1 = time()

# Convert to quaternions and assign as Quats
conv_quats = Quat(quat_conversion_torch.cubo2quat(rand_cubos, scalar_first=False)).to_torch()

# Preallocate ndarrays for symmetry and perform outer product
quat_syms = torch.tensor(0)
symcount = torch.tensor(0)
torch_hcp_syms, torch_fcc_syms = hcp_syms.to_torch(), fcc_syms.to_torch()

if symmetry_system is 'hcp':
    quat_syms = conv_quats.outer_prod(torch_hcp_syms)
    symcount = 12
else:
    quat_syms = conv_quats.outer_prod(torch_fcc_syms)
    symcount = 24

#####################
# Loss calculations with expanded quaternion symmetry would be calculated here using quat_syms
####################

# Create nadarray for Rodrigues expansion
rod_syms = torch.zeros((N, symcount, 4), dtype=torch.float64)

# Perform Rodrigues conversion on expanded symmetries
for current_sym in range(0, symcount):
    rod_syms[:, current_sym, :] = quat_conversion_torch.quat2rod(quat_syms.X[:, current_sym])

# For each symmetry expanded Quat, find index for lowest magnitude value of tan(omega/2)
rod_filter = torch.argmin((torch.abs(rod_syms)), 1)[:, -1]

# rod_filter is a 1d array of length N containing the index of the minimum magnitude
# Rodrigues vector for each set of symmetry expanded Quats.  This index corresponds to the
# second index value in rod_syms.  For example, if 2 HCP Quats are being symmetry expanded in
# Rodrigues space, rod_syms will be of size:
# [2(number of Quats), 12(number of symmetries), 4(number of Rodrigues terms)]
# and rod_filter will be of size [2(number of Quats)], and will contain the indices corresponding
# to which symmetry index (0-11) has the lowest magnitude value of tan(omega/2)

# Based on these indices, extract the min magnitude symmetry form for each quat

rod_reduced = rod_syms[torch.arange(0, N), rod_filter,:]

# Convert back to cubochoric space
cubo_reduced = quat_conversion_torch.rod2cubo(rod_reduced)

print("Conversion time for "+str(N)+" Samples = "+str(time()-t1)+" seconds")

