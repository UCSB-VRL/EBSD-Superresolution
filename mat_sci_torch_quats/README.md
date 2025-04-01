# mat_sci_torch_quats

A small codebase to compute a disorientation loss between two sets of orientations. In various applications, we would like for our models to output a crystal orientation, compare with a ground truth orientation, and backpropagate the loss. With the oritations represented as unit quaternions, misorientation is a straightforward distance calculation. However, crystal symmetries complicate this. A cubic lattice can be rotated 24 different ways, without changing the physical lattice output. Disorientation considers all possible symmetries for a crystal structure, and chooses the symmetry with the smallest distance.


## quats.py

The primary file for numpy and torch quaternions. Quats are stored in a "Quat" object. This is simply a wrapper for an ND array, with a length 4 quat vector as the last dimension. The Quat object should act like a numpy arry or torch tensor, but with the last layer hidden. All indexing and reshaping should work as if your array contains a quaternion in each element. Multiplying two quats defaults to hadamard/element-wise product, but outer product is also available. Rotation takes in an array of size (...,3), and rotates every point by each rotation in the quat matrix. If your quat array and points are the same size, or they are properly set up to broadcast, then you can use the element-wise option as well.

There are several distance functions as well:
 * quat_dist: Computes distance between two quaternions, along the sphere.
 * rot_dist: Computes distance between to rotation quaternions. Takes into account q <-> -q symmetry.
 * rot_dist_w_syms: Similar to rot_dist, but accepts a 1D array of symmetries to check by brute force.


## plotting_utils.py

A simple module which defines things like a unit cube in hexagonal closest pack. Running this script plots the unit cobe and a pair of axes for HCP and FCC.


## symmetries.py

Defines symmetries for HCP and FCC. Running this will plot all of the symmetries for each.


## plot_mis_orient_dists.py

Plots the misornientation distributions.


## check_backprop.py

Checks that backpropagation works. A good sample usage script to incorperate into a CNN.

