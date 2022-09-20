from quats import Quat, rand_quats

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 100


q = rand_quats(N)

R = q.X[:,1:] / q.X[:,0][:,None]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*R.T)

plt.show()





