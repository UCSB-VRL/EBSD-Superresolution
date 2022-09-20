import torch
import torch.nn as nn

class MisOrientation(nn.Module):
    """Misoreintation loss."""
    def __init__(self, args):
        super(MisOrientation, self).__init__()
        dist_type = args.dist_type
        act = args.act_loss
        syms_req = args.syms_req
        
        print(f'Parameters for Training Loss')
        print('+++++++++++++++++++++++++++++++++++++++++')
        print(f'dist_type: {dist_type}  activation:{act}  Symmetry:{syms_req}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        from mat_sci_torch_quats.losses import ActAndLoss, Loss
        from mat_sci_torch_quats.symmetries import hcp_syms

        if syms_req:
            syms = hcp_syms
        else:
            syms = None
        
        self.act_loss = ActAndLoss(act, Loss(dist_type, syms), quat_dim=1)
            
    def forward(self, sr, hr):
        loss = self.act_loss(sr, hr)
        loss = torch.mean(loss)
        
        return loss
