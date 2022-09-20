import quat_conversion_torch
from quats import Quat
from symmetries import hcp_syms, fcc_syms
from time import time
import torch
import math

# Generate random cubochoric coordinates

torchpi = torch.tensor(math.pi)**(2/3)
N = 100000
symmetry_system = 'hcp'

rand_cubos = (torchpi/2 - (-1*torchpi/2)) * torch.rand(N, 3) + (-1*torchpi/2)

t1 = time()

# Convert to quaternions and assign as Quats
conv_quats = Quat(quat_conversion_torch.cubo2quat(rand_cubos, scalar_first=True),scalar_first=True).to_torch()

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


# Maximum magnitude s value in quaternion vector is FZ vector
quat_filter = torch.argmax((torch.abs(quat_syms.X)), 1)[:, 0]

# Build filtered quaternions
quat_reduced = quat_syms.X[torch.arange(0, N), quat_filter, :]

# Since quaternions are redundant (q= -q), need to convert all quaternions to positive 4d hemisphere
# Element-wise checks inefficient.  Build matrix of all possible q and -q values and take max in |s|
quat_reduced_inverse = quat_reduced*-1
quat_abs_check = torch.cat((torch.unsqueeze(quat_reduced,-1),torch.unsqueeze(quat_reduced_inverse,-1)),-1)
# Find which value (q or -q) is the larger is s for each value in quat_reduced (larger will be always be positive)
quat_abs_filter = torch.argmax(quat_abs_check[:, 0, :], -1)
# Apply argmax filtering to quat_abs_check to redetermine FZ quaternions (all now positive)
quat_reduced = quat_abs_check[torch.arange(0,N), :, quat_abs_filter]
# Remove extra dimension used during concatenation for abs value check
quat_reduced = torch.squeeze(quat_reduced)

# Convert back to cubochoric (since scalar first was enforced for symmetry expansion, it should always be true here)
cubo_quat_reduced = quat_conversion_torch.quat2cubo(quat_reduced, scalar_first=True)  # Should always be true here

print("Conversion time for "+str(N)+" Samples = "+str(time()-t1)+" seconds")
