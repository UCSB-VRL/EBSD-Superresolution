import numpy as np
#from quats import Quat
import torch
from mat_sci_torch_quats.quats import outer_prod
#from quats import outer_prod

#import pdb; pdb.set_trace()
#rotate by 0 or 180 degrees about x axis
hcp_r1 = torch.eye(4)[:2]


# rotate about 0, 60, ... 300 degrees about z axis
hcp_r2 = torch.zeros((6,4))
hcp_r2[:,0] = torch.cos(torch.arange(6)/6*np.pi)
hcp_r2[:,3] = torch.sin(torch.arange(6)/6*np.pi)
hcp_syms = outer_prod(hcp_r1,hcp_r2).reshape((-1,4))

#hcp_syms = hcp_r2.outer_prod(hcp_r1).transpose((1,0)).reshape(-1)




if __name__ == '__main__':

    from plotting_utils import *

    np.random.seed(1)
    q = np.random.normal(0,1,4)
    q /= np.linalg.norm(q)
    q1 = Quat(q)

    rhomb_wire = path2prism(rhomb_path)
    all_rots = q1.outer_prod(hcp_syms)
    all_wires = all_rots.rotate(rhomb_wire)
    all_axes = all_rots.rotate(rhomb_axes)

    def setup_axes(m,n):
        r = np.sqrt(2)
        fig = plt.figure()
        axes = [fig.add_subplot(m,n,i+1,projection='3d') for i in range(m*n)]
        for a in axes:
            a.set_xlim(-r,r)
            a.set_ylim(-r,r)
            a.set_zlim(-r,r)    
        return fig, axes

    fig, axes = setup_axes(3,4)

    for i, ax in enumerate(axes):

        ax.plot(*all_wires[0].T,color='#888')
        ax.plot(*all_wires[i].T,color='#000')
        plot_axes(ax,all_axes[i])


    square_wire = path2prism(square_path)
    all_rots = q1.outer_prod(fcc_syms)
    all_wires = all_rots.rotate(square_wire)
    all_axes = all_rots.rotate(square_axes)

    fig, axes = setup_axes(4,6)

    for i, ax in enumerate(axes):
        ax.plot(*all_wires[0].T,color='#888')
        ax.plot(*all_wires[i].T,color='#000')
        plot_axes(ax,all_axes[i])


    plt.show()



