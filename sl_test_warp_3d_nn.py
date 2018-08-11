import numpy as np
# from numpy.testing import (assert_array_equal,
#                            assert_array_almost_equal,
#                            assert_almost_equal,
#                            assert_equal,
#                            assert_raises)
from scipy.ndimage.interpolation import map_coordinates
# from nibabel.affines import apply_affine, from_matvec
from dipy.core import geometry
from dipy.align import floating
# from dipy.align import imwarp
from dipy.align import vector_fields as vfu
# from dipy.align.transforms import regtransforms
# from dipy.align.parzenhist import sample_domain_regular

sh = (64, 64, 64)
ns = sh[0]
nr = sh[1]
nc = sh[2]

# Create an image of a sphere
radius = 24
sphere = vfu.create_sphere(ns, nr, nc, radius)
sphere = np.array(sphere, dtype=floating)

# Create a displacement field for warping
d, dinv = vfu.create_harmonic_fields_3d(ns, nr, nc, 0.2, 8)
d = np.asarray(d).astype(floating)

# Create grid coordinates
x_0 = np.asarray(range(sh[0]))
x_1 = np.asarray(range(sh[1]))
x_2 = np.asarray(range(sh[2]))
X = np.empty((4,) + sh, dtype=np.float64)
O = np.ones(sh)
X[0, ...] = x_0[:, None, None] * O
X[1, ...] = x_1[None, :, None] * O
X[2, ...] = x_2[None, None, :] * O
X[3, ...] = 1

# Select an arbitrary rotation axis
axis = np.array([.5, 2.0, 1.5])
# Select an arbitrary translation matrix
t = 0.1
trans = np.array([[1, 0, 0, -t * ns],
                  [0, 1, 0, -t * nr],
                  [0, 0, 1, -t * nc],
                  [0, 0, 0, 1]])
trans_inv = np.linalg.inv(trans)

# Select arbitrary rotation and scaling matrices
theta = np.pi / 5.0  # rotation angle
s = 1.1  # scale
rot = np.zeros(shape=(4, 4))
rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
rot[3, 3] = 1.0

scale = np.array([[1 * s, 0, 0, 0],
                  [0, 1 * s, 0, 0],
                  [0, 0, 1 * s, 0],
                  [0, 0, 0, 1]])

aff = trans_inv.dot(scale.dot(rot.dot(trans)))

# Select arbitrary (but different) grid-to-space transforms
sampling_grid2world = scale
field_grid2world = aff
field_world2grid = np.linalg.inv(field_grid2world)
image_grid2world = aff.dot(scale)
image_world2grid = np.linalg.inv(image_grid2world)

A = field_world2grid.dot(sampling_grid2world)
B = image_world2grid.dot(sampling_grid2world)
C = image_world2grid

# Reorient the displacement field according to its grid-to-space
# transform
dcopy = np.copy(d)
vfu.reorient_vector_field_3d(dcopy, field_grid2world)

extended_dcopy = np.zeros(
    (ns + 2, nr + 2, nc + 2, 3), dtype=floating)
extended_dcopy[1:ns + 1, 1:nr + 1, 1:nc + 1, :] = dcopy

# Compute the warping coordinates (see warp_2d documentation)
Y = np.apply_along_axis(A.dot, 0, X)[0:3, ...]
Z = np.zeros_like(X)
Z[0, ...] = map_coordinates(extended_dcopy[..., 0], Y + 1, order=1)
Z[1, ...] = map_coordinates(extended_dcopy[..., 1], Y + 1, order=1)
Z[2, ...] = map_coordinates(extended_dcopy[..., 2], Y + 1, order=1)
Z[3, ...] = 0
Z = np.apply_along_axis(C.dot, 0, Z)[0:3, ...]
T = np.apply_along_axis(B.dot, 0, X)[0:3, ...]
W = T + Z

# Test bilinear interpolation
# expected = map_coordinates(sphere, W, order=1)
# warped = vfu.warp_3d(sphere, dcopy, A, B, C,
#                      np.array(sh, dtype=np.int32))
# np.save('sl_syn_warp_3d.npy', warped)

# Test nearest neighbor interpolation
expected = map_coordinates(sphere, W, order=0)
warped = vfu.warp_3d_nn(sphere, dcopy, A, B, C,
                        np.array(sh, dtype=np.int32))
# assert_array_almost_equal(warped, expected, decimal=5)
np.save('sl_syn_warp_3d_nn.npy', warped)
