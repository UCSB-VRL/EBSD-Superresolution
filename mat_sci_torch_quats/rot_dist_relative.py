import torch
from quats import hadamard_prod, outer_prod, misorientation, inverse

class RotDistRelative(torch.nn.Module):
    # q_pred: ML output, q_gt: ground-truth (hr)
    def __init__(self, syms):
        self.syms = syms

    def forward(self, q_pred, q_gt):
        A = hadamard_prod(q_pred, inverse(q_gt))
        A_sym = outer_prod(A, self.syms)
        thetas = misorientation(A_sym, torch.Tensor([1,0,0,0])) # broadcasting
        min_theta = thetas.min(-1)[0]
        # A_min = A_sym[torch.arange(len(A_sym)), min_theta_inds] # symmetry reduce the transformation 
        # q2_min = inverse(hadamard_prod(inverse(q1), A_min)) # orientation from data2, towards sample (ipf convention)

        return min_theta

# class RotDistLoss(torch.nn.Module):
#     def forward(self,q_pred,q_gt):
#         #import pdb; pdb.set_trace()
#         q_pred_neg = torch.stack((q_pred,-q_pred),dim=-2)
#         if q_gt is not None: q_gt = q_gt[...,None,:]

#         euclid_dist = torch.linalg.norm(q_pred_neg-q_gt,dim=-1)
#         theta = EuclidToRotApprox()(euclid_dist)
#         theta_min = theta.min(-1)[0]
#         return theta_min

# class EuclidToRotApprox:
#     def __init__(self,beta=0.1,eps=0.01):
#         self.t = 2 - beta
#         self.eps = eps
#         t = torch.Tensor([self.t])
#         t.requires_grad = True
#         y = euclid2rot(t)
#         y.backward()
#         self.m = float(t.grad)
#         self.b = float(y) - self.m*self.t

#     def __call__(self,x):
#         x_clip = torch.clamp(x,self.eps,self.t)
#         #mask = (x < self.beta).float()
#         y_abs = torch.abs(x)
#         y_lin = self.m*x + self.b
#         y_rot = euclid2rot(x_clip)
#         y_out = y_abs * (x <= self.eps).float() + \
#                 y_rot * torch.logical_and(x > self.eps,x < self.t).float() + \
#                 y_lin * (x >= self.t).float()

#         return y_out

