
from mat_sci_torch_quats.quats import slerp_ref_symm

class Slerp:

  def __init__(self, lr, scale):
    self.lr = lr
    self.scale = scale

  def __call__(self):
    
