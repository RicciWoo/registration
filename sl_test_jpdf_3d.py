import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data

fetch_stanford_hardi()
nib_stanford, _ = read_stanford_hardi()
static = np.squeeze(nib_stanford.get_data())[..., 0]
static_grid2world = nib_stanford.affine

fetch_syn_data()
_, nib_syn_b0 = read_syn_data()
moving = np.array(nib_syn_b0.get_data())
moving_grid2world = nib_syn_b0.affine

static = np.array(static).astype(np.float64)
moving = np.array(moving).astype(np.float64)

import numpy.linalg as npl
moving_world2grid = npl.inv(moving_grid2world)

from dipy.align.imwarp import get_direction_and_spacings
dim = len(static.shape)
moving_direction, moving_spacing = \
    get_direction_and_spacings(moving_grid2world, dim)

from dipy.align.vector_fields import _gradient_3d
out_shape = static.shape
ftype = moving.dtype.type
out = np.empty(tuple(out_shape)+(dim,), dtype=ftype)
inside = np.empty(tuple(out_shape), dtype=np.int32)
_gradient_3d(moving, moving_world2grid, moving_spacing, 
             static_grid2world, out, inside)

mgrad = np.asarray(out)

from dipy.align.imaffine import AffineMap
dim = len(static.shape)
starting_affine = np.eye(dim + 1)
affine_map = AffineMap(starting_affine, static.shape, static_grid2world,
                       moving.shape, moving_grid2world)

static_values = static
moving_values = affine_map.transform(moving)

from dipy.align.transforms import AffineTransform3D
params = None
transform = AffineTransform3D()
static2prealigned = static_grid2world
histogram.update_gradient_dense(params, transform, static_values,
	                moving_values, static2prealigned, mgrad)

np.save('sl_aff_par_jpdf_joint_grad.npy', histogram.joint_grad)
