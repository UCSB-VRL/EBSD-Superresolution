
import math
from math import pi
import torch
import numpy as np


inv_sqrt_2 = 1 / torch.sqrt(torch.tensor([2], dtype=torch.float32))

half = 1 / torch.tensor([2], dtype=torch.float32)

# fcc_syms = torch.tensor([
#                 [1, 0, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1, 0],
#                 [0, 0, 0, 1],
#                 [inv_sqrt_2, inv_sqrt_2, 0, 0 ],
#                 [inv_sqrt_2, 0, inv_sqrt_2, 0],
#                 [inv_sqrt_2, 0, 0, inv_sqrt_2],
#                 [inv_sqrt_2, -inv_sqrt_2, 0, 0],
#                 [inv_sqrt_2, 0, -inv_sqrt_2, 0],
#                 [inv_sqrt_2, 0, 0, -inv_sqrt_2],
#                 [0, inv_sqrt_2, inv_sqrt_2, 0],
#                 [0, inv_sqrt_2, 0, inv_sqrt_2],
#                 [0, 0, inv_sqrt_2, inv_sqrt_2],
#                 [0, inv_sqrt_2, -inv_sqrt_2, 0],
#                 [0, 0, inv_sqrt_2, -inv_sqrt_2],
#                 [0, inv_sqrt_2, 0, -inv_sqrt_2],
#                 [half, half, half, half],
#                 [half, -half, -half, half],
#                 [half, -half, half, -half],
#                 [half, half, -half, -half],
#                 [half, half, half, -half],
#                 [half, half, -half, half],
#                 [half, -half, half, half],
#                 [half, -half, -half, -half],
#             ], dtype=torch.float32)

fcc_syms = torch.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [inv_sqrt_2, inv_sqrt_2, 0, 0 ],
                [inv_sqrt_2, 0, inv_sqrt_2, 0],
                [inv_sqrt_2, 0, 0, inv_sqrt_2],
                [inv_sqrt_2, -inv_sqrt_2, 0, 0],
                [inv_sqrt_2, 0, -inv_sqrt_2, 0],
                [inv_sqrt_2, 0, 0, -inv_sqrt_2],
                [0, inv_sqrt_2, inv_sqrt_2, 0],
                [0, inv_sqrt_2, 0, inv_sqrt_2],
                [0, 0, inv_sqrt_2, inv_sqrt_2],
                [0, inv_sqrt_2, -inv_sqrt_2, 0],
                [0, 0, inv_sqrt_2, -inv_sqrt_2],
                [0, inv_sqrt_2, 0, -inv_sqrt_2],
                [half, half, half, half],
                [half, -half, -half, half],
                [half, -half, half, -half],
                [half, half, -half, -half],
                [half, half, half, -half],
                [half, half, -half, half],
                [half, -half, half, half],
                [half, -half, -half, -half],

                # [-1, 0, 0, 0],
                # [0, -1, 0, 0],
                # [0, 0, -1, 0],
                # [0, 0, 0, -1],
                # [-inv_sqrt_2, -inv_sqrt_2, 0, 0],
                # [-inv_sqrt_2, 0, -inv_sqrt_2, 0],
                # [-inv_sqrt_2, 0, 0, -inv_sqrt_2],
                # [-inv_sqrt_2, inv_sqrt_2, 0, 0],
                # [-inv_sqrt_2, 0, inv_sqrt_2, 0],
                # [-inv_sqrt_2, 0, 0, inv_sqrt_2],
                # [0, -inv_sqrt_2, -inv_sqrt_2, 0],
                # [0, -inv_sqrt_2, 0, -inv_sqrt_2],
                # [0, 0, -inv_sqrt_2, -inv_sqrt_2],
                # [0, -inv_sqrt_2, inv_sqrt_2, 0],
                # [0, 0, -inv_sqrt_2, inv_sqrt_2],
                # [0, -inv_sqrt_2, 0, inv_sqrt_2],
                # [-half, -half, -half, -half],
                # [-half, half, half, -half],
                # [-half, half, -half, half],
                # [-half, -half, half, half],
                # [-half, -half, -half, half],
                # [-half, -half, half, -half],
                # [-half, half, -half, -half],
                # [-half, half, half, half]
               
            ], dtype=torch.float32)



# Defines mapping from quat vector to matrix. Though there are many
# possible matrix representations, this one is selected since the
# first row, X[...,0], is the vector form.
# https://en.wikipedia.org/wiki/Quaternion#Matrix_representations
q1 = np.diag([1,1,1,1])
qj = np.roll(np.diag([-1,1,1,-1]),-2,axis=1)
qk = np.diag([-1,-1,1,1])[:,::-1]
qi = np.matmul(qj,qk)
Q_arr = torch.Tensor([q1,qi,qj,qk])
Q_arr_flat = Q_arr.reshape((4,16))

# Checks if 2 arrays can be broadcast together
def _broadcastable(s1,s2):
        if len(s1) != len(s2): return False
        else: return all((i==j) or (i==1) or (j==1) for i,j in zip(s1,s2))

# Converts an array of quats as vectors to matrices. Generally
# used to facilitate quat multiplication.
def vec2mat(X):
        assert X.shape[-1] == 4, 'Last dimension must be of size 4'
        new_shape = X.shape[:-1] + (4,4)
        dtype = X.dtype
        Q = Q_arr_flat.type(X.dtype).to(X.device)
        #print('Q', Q.dtype)
        return torch.matmul(X,Q).reshape(new_shape)

# Used if quaternion is in vector form
def hamilton_prod(q1,q2):

        # import pdb; pdb.set_trace()

        if (q1.shape[1] > q2.shape[1]): # one of them will have the interpolation parameter t embedded in that dimension
                q_new = torch.zeros(q1.shape)
        else:
                q_new = torch.zeros(q2.shape)

        q_new[...,0] = q1[...,0]*q2[...,0] - q1[...,1]*q2[...,1] - q1[...,2]*q2[...,2] - q1[...,3]*q2[...,3]
        q_new[...,1] = q1[...,0]*q2[...,1] + q1[...,1]*q2[...,0] + q1[...,2]*q2[...,3] - q1[...,3]*q2[...,2]
        q_new[...,2] = q1[...,0]*q2[...,2] - q1[...,1]*q2[...,3] + q1[...,2]*q2[...,0] + q1[...,3]*q2[...,1]
        q_new[...,3] = q1[...,0]*q2[...,3] + q1[...,1]*q2[...,2] - q1[...,2]*q2[...,1] + q1[...,3]*q2[...,0]

        return q_new

# quaternion (4x4) matrix-based hamilton product
def matrix_hamilton_prod(q1,q2):

        # import pdb; pdb.set_trace()

        assert _broadcastable(q1.shape,q2.shape), 'Inputs of shapes ' \
                        f'{q1.shape}, {q2.shape} could not be broadcast together'
        X1 = vec2mat(q1)
        X_out = (X1 * q2[...,None,:]).sum(-1)
        return X_out

# Performs outer product on ndarrays of quats
# Ex if X1.shape = (s1,s2,4) and X2.shape = (s3,s4,s5,4),
# output will be of size (s1,s2,s3,s4,s5,4)
# note!: useful function for efficent (gpu parallelized) implementation of slerp
def outer_prod(q1,q2):
        X1 = vec2mat(q1)
        X2 = torch.movedim(q2,-1,0)
        X1_flat = X1.reshape((-1,4))
        X2_flat = X2.reshape((4,-1))
 
        X_out = torch.matmul(X1_flat,X2_flat)
        X_out = X_out.reshape(q1.shape + q2.shape[:-1])
        X_out = torch.movedim(X_out,len(q1.shape)-1,-1)
        return X_out

# Utilities to create random vectors on the L2 sphere. First produces
# random samples from a rotationally invariantt distibution (i.e. Gaussian)
# and then normalizes onto the unit sphere

# Produces random array of the same size as shape.
def rand_arr(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        X = torch.randn(shape).type(dtype)
        X /= torch.norm(X,dim=-1,keepdim=True)
        return X

# Produces array of 3D points on the unit sphere.
def rand_points(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape + (3,), dtype)

# Produces random unit quaternions.
def rand_quats(shape,dtype=torch.FloatTensor):
        if not isinstance(shape,tuple): shape = (shape,)
        return rand_arr(shape+(4,), dtype)

# arccos, expanded from range [-1,1] to all real numbers
# values outside of [-1,1] and replaced with a line of slope pi/2, such that
# the function is continuous
# this is relevant for ML
def safe_arccos(x):
    mask = (torch.abs(x) < 1).float()
    x_clip = torch.clamp(x,min=-1,max=1)
    output_arccos = torch.arccos(x_clip)
    output_linear = (1 - x)*pi/2
    output = mask*output_arccos + (1-mask)*output_linear
    return output

def quat_dist(q1,q2=None):
        """
        Computes distance between two quats. If q1 and q2 are on the unit sphere,
        this will return the arc length along the sphere. For points within the
        sphere, it reduces to a function of MSE.
        """
        # import pdb; pdb.set_trace()
        if q2 is None: mse = (q1[...,0]-1)**2 + (q1[...,1:]**2).sum(-1)
        else: mse = ((q1-q2)**2).sum(-1)
        
        corr = 1 - (1/2)*mse
        corr_clamp = torch.clamp(corr,-1,1)
        return safe_arccos(corr)

# function below doesn't work if we cross a symmetry line
def misorientation(q1, q2):
#   import pdb; pdb.set_trace()
  return(4*torch.arcsin(torch.clamp(torch.linalg.vector_norm(q1 - q2, 2, dim=-1)/2,min=-1,max=1))) 

def misor(q1, q2):
#        import pdb; pdb.set_trace()
#        q2_repeat = q2[None,...]
#        q2_repeat = q2_repeat.repeat(5,1)
       q_diff = matrix_hamilton_prod(q1, inverse2(q2))
       return 2*torch.arccos(q_diff[...,0])

# Returns a single minimum theta, with 24x24x1 symmetry
def symm_double_misorientation(q1, q2):

        import pdb; pdb.set_trace()

        q1_w_syms = outer_prod(q1, fcc_syms)
        q2_w_syms = outer_prod(q2, fcc_syms)

        syms_diff_matrix = torch.zeros(max(q1.shape[0], q2.shape[0]),24,24)

        for i in range(24):
                for j in range(24):
                        syms_diff_matrix[:,i,j] = misorientation(q1_w_syms[:,i,:], q2_w_syms[j,:]) # broadcast misorientation calc across the symmetry dimension.
        import pdb; pdb.set_trace()

        theta_mins = torch.Tensor([min(syms_diff_matrix[i].view(-1)) for i in range(5)])

        return theta_mins

# Returns a single minimum theta, with 24x24x2 symmetry
def symm_complete_misorientation(q1, q2):
        # import pdb; pdb.set_trace()

        q1_w_syms = outer_prod(q1, fcc_syms)
        q2_w_syms = outer_prod(q2, fcc_syms)

        syms_diff_matrix = torch.zeros(max(q1.shape[0], q2.shape[0]), 24, 24, 2)

        for i in range(24):
               for j in range(24):
                      syms_diff_matrix[:,i,j,0] = misor(q1_w_syms[:,i,:], q2_w_syms[None,j,:].repeat(5,1))
                      syms_diff_matrix[:,i,j,1] = misor(q2_w_syms[None,j,:].repeat(5,1), q1_w_syms[:,i,:])

        import pdb; pdb.set_trace()
        theta_mins = torch.Tensor([min(syms_diff_matrix[i].view(-1)) for i in range(5)])
        return theta_mins






        # remove theta-0's (quaternion being compared to itself).
        # syms_diff_matrix = syms_diff_matrix[syms_diff_matrix != 0]
        # return torch.min(syms_diff_matrix.view(-1)) # find absolute minimum of multi-dimensional tensor by flattening it
        # quats_num = max(q1.shape[0], q2.shape[0])
        # min_thetas = torch.zeros(quats_num)

        # for i in range(quats_num):
        #         min_thetas[i] = torch.min(syms_diff_matrix[i],-1)

        # Find the minimum value and its index
        # min_value, min_index_flat = torch.min(syms_diff_matrix, dim=None)

        # # (Will be useful for selecting q1 and q2 for slerp) Convert the overall minimum index to row and column indices
        # min_index_row = min_index_flat // syms_diff_matrix.size(1)  # integer division for row index
        # min_index_col = min_index_flat % syms_diff_matrix.size(1)   # remainder for column index



def symm_misorientation(q1, q2=None):

  import pdb; pdb.set_trace()
  if (q2 is None):
    q2 = torch.Tensor([1,0,0,0])

  # delete this after!!!
#   q1 = scalar_first2last(q1)
#   q2 = scalar_first2last(q2)

  q1_w_syms = outer_prod(q1, fcc_syms)
  q2 = q2.repeat(tuple(q1_w_syms.shape[:-1]) + tuple([1]))   

  # I think outer product is just computing misorientation between each pair of same iterations of the 24 symmetries (i.e., Not all possible permutations)

  # import pdb; pdb.set_trace()
  dists = misorientation(q1_w_syms, q2)
  disorientation = torch.min(dists,-1)[0]

  return(disorientation)
        
def rot_dist(q1,q2=None):
        """ Get dist between two rotations, with q <-> -q symmetry """
        # import pdb; pdb.set_trace()
        # q1_w_neg = torch.stack((q1,-q1),dim=-2)
        if q2 is not None: q2 = q2[...,None,:]
        dists = quat_dist(q1,q2)
        dist_min_index = dists.min(-1)[1]
        return dist_min_index

def fz_reduce(q,syms):
        shape = q.shape
        q = q.reshape((-1,4))
        # syms = syms.cuda()
        q_w_syms = outer_prod(q,syms)
        # import pdb; pdb.set_trace
        dists = rot_dist(q_w_syms)
        inds = dists.min(-1)[1]
        q_fz = q_w_syms[torch.arange(len(q_w_syms)),dists]
        q_fz *= torch.sign(q_fz[...,:1])
        q_fz = q_fz.reshape(shape)
        return q_fz

def inverse2(q):
        # import pdb; pdb.set_trace()
        q = torch.Tensor([1,-1,-1,-1])*q
        norm = (torch.linalg.vector_norm(q, 2, dim=-1))**2
        mask = (norm != 0) # mask avoids dividing by zero-norm, if q = [0,0,0,0]

        temp = q[mask].t() / norm[mask] 
        q[mask] = temp.t()
        # q = q.t() / norm
        return q

# quaternion 'q', to the power of 't'
# you need to understand 
def quat_exp2(q, t):

        # import pdb; pdb.set_trace()

        # Quaternion normalization
        mag = torch.linalg.vector_norm(q,2,-1).unsqueeze(-1)
        mask1 = (torch.all(q != torch.Tensor([0,0,0,0]),dim=-1))
        q[mask1] = q[mask1] / mag[mask1] # use mask to avoid dividing by zero
        phi = torch.arccos(torch.clamp(q[:,0],min=-1,max=1))

        # Versor normalization
        v = q[:,1:4]
        v_unit = v 
        mask2 = (torch.all(q[:,1:4] != torch.Tensor([0,0,0]),dim=-1))

        # obtain unit versor (not present in slerp3 code)
        # q[:, 1:4] is not normalized, this should be equivalent to dividing it by sin(theta/2)
        v_unit[mask2] = v[mask2] / torch.linalg.vector_norm(v[mask2],2,-1).unsqueeze(-1)

        # qm.w = (qa.w * 0.5 + qb.w * 0.5);
	# 	qm.x = (qa.x * 0.5 + qb.x * 0.5);
	# 	qm.y = (qa.y * 0.5 + qb.y * 0.5);
	# 	qm.z = (qa.z * 0.5 + qb.z * 0.5);

        slerp_angles = torch.outer(phi, t) # size=[7372, 3]: [# of quats, # of interpolation parameters]
        cos_slerp = torch.cos(slerp_angles) # size=[7372,3]
        sin_slerp = torch.sin(slerp_angles) # size=[7372,3]

        q_new = torch.zeros([q.shape[0], t.shape[0], 4])

        q_new[:,:,0] = cos_slerp
        q_new[:,:,1:4] = v_unit.unsqueeze(1) * sin_slerp.unsqueeze(-1) 

        return q_new

# quaternion to a scalar power
def quat_exp(q, t):

        # import pdb; pdb.set_trace()
        theta = math.acos(np.clip(q[0],-1,1))

        v = q[1:4]
        norm = torch.linalg.norm(q[1:4],2)

        if (v.all() != 0):
                v = q[1:4] / norm

        return torch.Tensor([math.cos(theta*t),v[0].item()*math.sin(theta*t),v[1].item()*math.sin(theta*t),v[2].item()*math.sin(theta*t)])

# Single quaternion slerp calculation
def slerp_calc(q1, q2, t):
        # import pdb; pdb.set_trace()
        q_slerp = matrix_hamilton_prod(q1, quat_exp(matrix_hamilton_prod(inverse2(q1),q2), t))
        return q_slerp

# Parallel slerp calculation
def slerp_calc2(q1, q2, t):
        # import pdb; pdb.set_trace()
        # edited to unsqueeze q1 in dim=1, to render it broadcastable with the exponentiated quaternion for various values of interpolation parameter 't'

        # if q2 = None, we want to slerp only with respect to the axis formed by the first quaternion with respect to Theta = 0.
        if (q2 is None):
                q2 = torch.Tensor([1,0,0,0])
                q2 = q2[None, None, :]
                q2 = torch.repeat(q1.shape[0], q1.shape[1], 1)

        # choose q2 that produces an angle smaller than 90 degrees.
        q_slerp = hamilton_prod(quat_exp2(hamilton_prod(q2, inverse2(q1)), t), q1.unsqueeze(1).repeat(1,3,1))

        return q_slerp

def slerp_calc3(q1, q2, t):
       A = matrix_hamilton_prod(inverse2(q1), q2)
       A_syms = outer_prod(A, fcc_syms)
       A_syms *= torch.sign

       qs = inverse2(inverse2(q1))

# should only need to apply disorientation slerp twice, once to fill in cols, and once to fill in rows.
# used for symmetry-aware-slerp: compare all slerps
        # inputs required: symmetry matrix, quaternions whose misorientations are being compared.

def disorientation_slerp(q1, q2, t, syms):

        # import pdb; pdb.set_trace()
        q1_w_syms = outer_prod(q1, syms)

        # q2_repeat = q2.unsqueeze(1)
        # q2_repeat = q2_repeat.repeat(1, 48, 1)

        dists = misorientation(q1_w_syms, q2[None,:])
        inds1 = torch.min(dists,-1)[1]

        # q1_min = q1_w_syms[torch.arange(len(q1_w_syms)), inds1]
        q1_min = q1_w_syms[inds1, :]

        q3 = slerp_calc(q1_min, q2, t) # only one slerp calculated, per thread, for two quats with minimum disorientation

        # Set to rotation=0 w respect to reference, if the symm_misorientation is not texture
        # if (symm_misorientation(q1,q3)*180/np.pi > 5):
        #        q3 = torch.Tensor([1,0,0,0])
      
        # if symm_misorientation between data points is smaller than 5, we should be in a grain region.
        # if (symm_misorientation(q1_min,q2)*180/np.pi < 5 and symm_misorientation(q1_min,q3)*180/np.pi > 5):
        #         import pdb; pdb.set_trace()

        # if (symm_misorientation(q1_min,q2)*180/np.pi < 5 and symm_misorientation(q2, q3)*180/np.pi > 5):
        #         import pdb; pdb.set_trace()

        return q3

        # X_scaled[indices[0], indices[1], :] = q3


# def slerp(X_scaled, indices, q1, q2, t, syms):

#         # import pdb; pdb.set_trace()
#         q1_w_syms = outer_prod(q1, syms)

#         q2_repeat = q2.unsqueeze(1)
#         q2_repeat = q2_repeat.repeat(1, 48, 1)

#         dists = misorientation(q1_w_syms, q2_repeat)
#         inds1 = torch.min(dists,1)[1]

#         q1_min = q1_w_syms[torch.arange(len(q1_w_syms)), inds1]

#         q3 = slerp_calc2(q1_min, q2, t) # only one slerp calculated, per thread, for two quats with minimum disorientation

#         X_scaled[indices[0], indices[1], :] = q3
 
 

# num_syms parameter: should disorientation be calculated for no quaternions, 1 quaternion, or both (2) quaternions
def slerp2(q1, q2, t, syms, num_syms=0):

        if (num_syms == -3):
                import pdb; pdb.set_trace()
                A = matrix_hamilton_prod(q1, inverse2(q2)) # transformation from crystal frame 1, to crystal frame 2
                A_syms = outer_prod(A, fcc_syms) # symmetry equivalent transformations
                A_syms *= torch.sign(A_syms[...,:1]) # ensure we are on 
                a_min_indices = torch.max(A_syms[...,0],-1)[1]
                A_min = A_syms[torch.arange(len(A_syms)), a_min_indices]
                qs = inverse2(matrix_hamilton_prod(inverse2(q1), A_min))
                
                q3 = slerp_calc2(q1, qs, t)
                return q3



        # avoid symmetry (fundamental zone) crossing:
        if (num_syms == -2):
                q1_w_syms = outer_prod(q1, syms)
                import pdb; pdb.set_trace()
                q1_diff = torch.sum(torch.abs(q1_w_syms[...,1:4] - q2[:, None, 1:4]), -1)
                inds = torch.min(q1_diff, -1)[1]
                q1_symm_restrict = q1_w_syms[torch.arange(len(q1_w_syms)), inds]

                q3 = slerp_calc2(q1_symm_restrict, q2, t)

                return q3
        
        # Attempting pair-wise symmetry now.
        if (num_syms == -1):
                import pdb; pdb.set_trace()

                # apply only the same symmetry operators to q1 and q2.

                q1_w_syms = outer_prod(q1, syms)
                q2_w_syms = outer_prod(q2, syms)

                dists1 = misorientation(q1_w_syms, q2[:,None,:])
                dists2 = misorientation(q2_w_syms, q1[:,None,:])

                [theta1_min, inds1] = torch.min(dists1, -1)
                [theta2_min, inds2] = torch.min(dists2, -1)

                q1_min = q1_w_syms[torch.arange(len(q1_w_syms)), inds1]
                q2_min = q2_w_syms[torch.arange(len(q2_w_syms)), inds2]
                q3A = slerp_calc2(q1_min, q2, t)
                q3B = slerp_calc2(q1, q2_min, t)

                mask = (theta1_min < theta2_min)
                q3 = q3B
                q3[mask] = q3A[mask]

      
                # thetas = misorientation(q1[:,None,:], q3)
                # long_path_mask = (torch.abs(thetas) > np.pi/2)
                # q3[long_path_mask] = slerp_calc2(q1, -q2, t)[long_path_mask]

                return q3

                # q3 = slerp_calc2(q1, q2, t)
                # q3_w_syms = outer_prod(q3,syms)

                # for i in range(3):
                #         q = q3_w_syms[:,i,:,:]
                #         dists1 = rot_dist(q, q1)
                #         dists2 = rot_dist(q, q2)
                #         dists_net = torch.abs(dists1 - dists2) / (dists1)
                #         inds = torch.min(dists_net,-1)[1]
                #         q3[:,i,:] = q[torch.arange(len(q)), inds]

                # return q3
                

        # symmetry reduction of interpolated point vs q1 or q2
        if (num_syms == 0):

                import pdb; pdb.set_trace()

                q3 = slerp_calc2(q1, q2, t)
                # choose q2 that produces an angle smaller than 90 degrees.
                thetas = misorientation(q1[:,None,:], q3)
                long_path_mask = (torch.abs(thetas) > np.pi/2)
                q3[long_path_mask] = slerp_calc2(q1, -q2, t)[long_path_mask]


                # re-orient interpolated q3 with respect to original q1
                
                # q3 = slerp_calc2(q3, q1_repeat)

                # q3_w_syms = outer_prod(q3,syms)
                # # q1 = q1[:,None,:]
                # # q1_repeat = q1.repeat(1,48,1)

                # for i in range(3):
                #         q = q3_w_syms[:,i,:,:]
                #         inds = rot_dist(q, q2)
                #         q3[:,i,:] = q[torch.arange(len(q)), inds]

                return q3

        # q1_w_syms = outer_prod(q1, syms)

        # q2_repeat = q2.unsqueeze(1)
        # q2_repeat = q2_repeat.repeat(1, 48, 1)
        # # dists = misorientation(q1_w_syms, q2_repeat)
        # dists = quat_dist(q1_w_syms, q2_repeat)
        
        # # remove nan from theta tensor (dists)

        # inds1 = torch.min(dists,-1)[1]
        # q1_min = q1_w_syms[torch.arange(len(q1_w_syms)), inds1]

        if(num_syms == 1):
                # import pdb; pdb.set_trace()

                q1_w_syms = outer_prod(q1, syms)
                q2_repeat = q2[:,None,:].repeat(1,24,1)
                dists1 = misorientation(q1_w_syms, q2_repeat)
                [theta1_min, inds1] = torch.min(dists1, -1)

                q1_min = q1_w_syms[torch.arange(len(q1_w_syms)), inds1]
                q3 = slerp_calc2(q1_min, q2, t)
         

                # re-orient interpolated q3 with respect to original q1.


                return q3

        if(num_syms == 2):

        # import pdb; pdb.set_trace()

                q2_w_syms = outer_prod(q2, syms)
                q2_syms_permutations = q2_w_syms.unsqueeze(2)
                q2_syms_permutations = q2_syms_permutations.repeat(1,1,48,1)

                min_theta = 1000*torch.ones(q1.shape[0])
                indices1 = torch.zeros(q1.shape[0]) # indices for Q1 matrix
                indices2 = torch.zeros(q2.shape[0]) # indices for Q2 matrix

                q1_min = torch.zeros(q1.shape[0], 4)
                q2_min = torch.zeros(q2.shape[0], 4)

                for i in range(48):

                        # import pdb; pdb.set_trace()
                        # checking theta vs quat_dist performance
                        dists = misorientation(q1_w_syms, q2_syms_permutations[:,i,i,:].unsqueeze(1)) # (7000, 48, 4) vs. (7000, 4) --> second one should get broadcasted.
                        dists = quat_dist(q1_w_syms, q2_syms_permutations[:,i,i,:].unsqueeze(1)) 

                        # dists here is 7000 by 48 (good.)
                        theta = torch.min(dists,-1)[0] # 
                        inds = torch.min(dists,-1)[1] # provides 7000 indices of range(0,48), to select symmetry w minimum theta, for each datapoint.                                                                                                

                        min_mask = (theta < min_theta)
                        min_theta[min_mask] = theta[min_mask] # update minimum theta for points where theta < minimum_theta.

                        q2_min[min_mask] = q2_syms_permutations[min_mask][:,i,i,:]
                        # q1_temp = q1_w_syms[inds]

                        q1_temp = q1_w_syms[torch.arange(len(q1_w_syms)), inds]

                        q1_min[min_mask] = q1_temp[min_mask]

                q3 = slerp_calc2(q1_min, q2_min, t)

                return q3

        if(num_syms == 3):
                q3 = slerp_calc2(q1, q2, t)
                q3_w_syms = outer_prod(q3, syms)
                q3_w_syms = torch.movedim(q3_w_syms,2,1)
                dists = misorientation(q1[:,None,None,:], q3) # unsqueeze two middle dimensions for q1.
                inds = torch.min(dists,-1)[1]
                q3 = q3_w_syms[torch.arange(len(q3_w_syms)), inds]

                return q3




        # dists = misorientation(q1_syms_permutations, q2_syms_permutations)
        # inds = torch.min(dists,-1)[1]
        # q1_min = q1_w_syms[torch.arange(len(q1_w_syms)), inds1]
        # q2_temp = q2_syms_permutations[:,:,permutation_index,:]
        # q2_min = q2_temp[torch.arange(len(q2_temp)), inds1]

        # q3 = slerp_calc2(q1_min, q2_min, t)
        # return q3

        # q2_w_syms = outer_prod(q2, syms)
        # dists = misorientation(q1_w_syms, q2_w_syms)

        # inds = torch.min(dists,-1)[1]
        # q1_min = q2_w_syms[torch.arange(len(q1_w_syms)), inds]
        # q2_min = q2_w_syms[torch.arange(len(q2_w_syms)), inds]

        # q3 = slerp_calc2(q1_min, q2, t)
        # q3_w_syms = torch.outer(q3, syms)

        # dists = misorientation(q3_w_syms, q1)
        # inds = torch.min(dists,-1)[1]

        # q3_min = q3_w_syms[torch.arange(len(q3_w_syms)), inds]

        # indices_permute = list(permutations(range(48)))

        # q1_w_syms_permute = torch.zeros(q1_syms.shape[0], 48*48, 4)
        # for i, indices in enumerate(indices_permute):
        #         q1_w_syms_permute[:,i,:] = q1_w_syms[:,indices, :]

        # return q3_min


def quat_upsampling_symm3(X,scale=4):

        device = torch.device('cuda:0')

        interp_rows = (scale-1)*(X.shape[0]-1)
        interp_cols = (scale-1)*(X.shape[1]-1)

        X_scaled = torch.zeros((scale*X.shape[0], scale*X.shape[1], 4), dtype=X.dtype)
        X_scaled[::scale, ::scale, :] = X

        delta_t = 1 / scale
        t = torch.Tensor([delta_t * k for k in range(1, scale)]) # try to make 't' outer product friendly
        q1 = torch.Tensor([]); q2 = torch.Tensor([])

        # perform slerp column-wise
        for j in range(X.shape[1]-1):
                q1 = torch.cat([q1, X[:,j,:]])
                q2 = torch.cat([q2, X[:,j+1,:]])
        
        q1.to(device); q2.to(device); t.to(device)

        q_interp_cols = slerp2(q1, q2, t, fcc_syms, -3) # seems like there are no NaN values here.
        q_interp_cols = q_interp_cols.reshape(-1,X.shape[0],3,4)
        # dimensions should be the amount of interpolation pairs, rows, interpolations per pair, 4

        for i in range(q_interp_cols.shape[0]): # looping through all interpolations pairs
                indices2 = range(scale*i + 1, scale*i+scale, 1)
                X_scaled[::scale, indices2, :] = q_interp_cols[i,:,:,:]
    
        # Now, perform row-based interplation
        q1 = torch.Tensor([]); q2 = torch.Tensor([])
        for i in range(X.shape[0]-1):
                q1 = torch.cat([q1, X_scaled[i*scale,:,:]])
                q2 = torch.cat([q2, X_scaled[(i+1)*scale,:,:]])

        q_interp_rows = slerp2(q1, q2, t, fcc_syms, -3)
        q_interp_rows = q_interp_rows.reshape(-1,X_scaled.shape[1],3,4)
        q_interp_rows = torch.movedim(q_interp_rows, 2, 1)

        for i in range(q_interp_rows.shape[0]): # looping through all interpolations pairs
                indices2 = range(scale*i + 1, scale*i+scale, 1)
                X_scaled[indices2, :, :] = q_interp_rows[i,:,:,:]
        # X_scaled[::scale, ::scale, :] = torch.zeros(4) # return original data points to zero, to ipf map only interpolated points.

        return X_scaled

# Second upsampling function attempt, with parallel computing, but nested for loops than necessary
def quat_upsampling_symm2(X,scale=4):

        device = torch.device('cuda:0')
        X_scaled = torch.zeros((scale*X.shape[0], scale*X.shape[1], 4))
        # place non-interpolated values
        # for i in range(X.shape[0]):
        #         for j in range(X.shape[1]):
        #                 X_scaled[i*scale, j*scale, :] = X[i, j, :]

        X_scaled[::scale, ::scale, :] = X 

        delta_t = 1 / scale

        q1 = torch.Tensor([]); q2 = torch.Tensor([]); t = torch.Tensor([]); indices = []

        # load column-wise interpolations
        for i in range(X.shape[0]):
                for j in range(X.shape[1]-1):
                        for k in range(1,scale):
                                q1 = torch.cat([q1, X[i][j]])
                                q2 = torch.cat([q2, X[i][j+1]])
                                t = torch.cat((t, torch.tensor([delta_t * (k+1)])))
                                indices.append((i*scale, j*scale + k)) # where to place interpolated values within the upsampled matrix
        q1.to(device); q2.to(device); t.to(device)
        #  perform column-wise interpolations, parallely on gpu. Note that torch Tensors are mutable objects, meaning they are passed by reference in python.
        slerp(X_scaled, indices, q1, q2, t, fcc_syms)

        q1 = torch.Tensor([]); q2 = torch.Tensor([]); t = torch.Tensor([]); indices = []

        # load row-wise interpolations
        for i in range(X.shape[0]-1):
                for j in range(X_scaled.shape[1]):
                        for k in range(1,scale):
                                q1 = torch.cat([q1, X_scaled[i*scale][j]])
                                q2 = torch.cat([q2, X_scaled[(i+1)*scale][j]])
                                t = torch.cat((t, torch.Tensor([delta_t * (k+1)])))
                                indices.append((i*scale + k, j))
        q1.to(device); q2.to(device); t.to(device)
        slerp(X_scaled, indices, q1, q2, t, fcc_syms)

        return(X_scaled)

# Original function used for upsampling, without parallel computing
def quat_upsampling_symm(X, scale=4, symm=False): # X is low-resolution numpy-array
        # import pdb; pdb.set_trace()

        # X = scalar_last2first(X) # convert from quaternion scalar_last convention to scalar_first.
        X_rows = X.shape[0]; X_cols = X.shape[1]
        interp_factor = scale
        X_SR = np.zeros([scale * X_rows, scale * X_cols, 4], dtype=np.float32) # Super-resolved numpy-array

        for i in range(scale * X_rows):

            for j in range(scale * X_cols):

                # import pdb; pdb.set_trace()

                x1 = math.floor(j/scale)
                x2 = math.floor(j/scale) + 1
                y1 = math.floor(i/scale)
                y2 = math.floor(i/scale) + 1

                # Check algorithm didn't exit bounds of original quat matrix
                if (y2 == X.shape[0]):
                    y2 = X.shape[0] - 1
                if (x2 == X.shape[1]):
                    x2 = X.shape[1] - 1

                q1 = X[y2][x1] # quats from LR cell
                q2 = X[y2][x2]
                q3 = X[y1][x1]
                q4 = X[y1][x2]
                t_x = (j % scale) / scale # find mod, and normalize to obtain interpolation parameter, 't'.
                
                if (symm == True):
                        q_interp_x1 = disorientation_slerp(q1, q2, t_x, fcc_syms)
                        q_interp_x2 = disorientation_slerp(q3, q4, t_x, fcc_syms)
                else:
                        q_interp_x1 = slerp_calc(q1, q2, t_x)
                        q_interp_x2 = slerp_calc(q3, q4, t_x)

                # y-interpolation:
                t_y = (i % scale) / scale

                if (symm == True):
                        X_SR[i][j] = disorientation_slerp(q_interp_x1, q_interp_x2, t_y, fcc_syms) # slerp returns Quat object, so get its numpy array.
                else:
                        X_SR[i][j] = slerp_calc(q_interp_x1, q_interp_x2, t_y) # slerp returns Quat object, so get its numpy array.

        return X_SR 

def scalar_first2last(X):
        return torch.roll(X,-1,-1)

def scalar_last2first(X):
        return torch.roll(X,1,-1)

def conj(q):
        q_out = q.clone()
        q_out[...,1:] *= -1
        return q_out


def rotate(q,points,element_wise=False):
        points = torch.as_tensor(points)
        P = torch.zeros(points.shape[:-1] + (4,),dtype=q.dtype,device=q.device)
        assert points.shape[-1] == 3, 'Last dimension must be of size 3'
        P[...,1:] = points
        if element_wise:
                X_int = hadamard_prod(q,P)
                X_out = hadamard_prod(X_int,conj(q))
        else:
                X_int = outer_prod(q,P)
                inds = (slice(None),)*(len(q.shape)-1) + \
                                (None,)*(len(P.shape)) + (slice(None),)
                X_out = (vec2mat(X_int) * conj(q)[inds]).sum(-1)
        return X_out[...,1:]


# A simple script to test the quats class for numpy and torch
if __name__ == '__main__':

        np.random.seed(1)
        N = 700
        M = 1000
        K = 13

        def test(dtype,device):

                q1 = rand_quats(M,dtype).to(device)
                q2 = rand_quats(N,dtype).to(device)
                q3 = rand_quats(M,dtype).to(device)
                p1 = rand_points(K,dtype).to(device)

                p2 = rotate(q2,rotate(q1,p1))
                p3 = rotate(outer_prod(q2,q1),p1)
                p4 = rotate(conj(q1[:,None]),rotate(q1,p1),element_wise=True)

                print('Composition of rotation error:')
                err = abs(p2-p3).sum()/len(p2.reshape(-1))
                print('\t',err)

                print('Rotate then apply inverse rotation error:')
                err = abs(p4-p1).sum()/len(p1.reshape(-1))
                print('\t',err,'\n')

        
        print('CPU Float 32')
        test(torch.cuda.FloatTensor,'cpu')

        print('CPU Float64')
        test(torch.cuda.DoubleTensor,'cpu')     

        if torch.cuda.is_available():

                print('CUDA Float 32')
                test(torch.cuda.FloatTensor,'cuda')

                print('CUDA Float64')
                test(torch.cuda.DoubleTensor,'cuda') 

        else:
                print('No CUDA')

