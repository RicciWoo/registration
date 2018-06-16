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

static = static[:, :, 37]
moving = moving[:, :, 37]
static_grid2world = np.delete(static_grid2world, 2, 0)
static_grid2world = np.delete(static_grid2world, 2, 1)
moving_grid2world = np.delete(moving_grid2world, 2, 0)
moving_grid2world = np.delete(moving_grid2world, 2, 1)

static = np.array(static).astype(np.float64)
moving = np.array(moving).astype(np.float64)

import numpy.linalg as npl
moving_world2grid = npl.inv(moving_grid2world)

from dipy.align.imwarp import get_direction_and_spacings
dim = len(static.shape)
moving_direction, moving_spacing = \
    get_direction_and_spacings(moving_grid2world, dim)

from dipy.align.vector_fields import _gradient_2d
out_shape = static.shape
ftype = moving.dtype.type
out = np.empty(tuple(out_shape)+(dim,), dtype=ftype)
inside = np.empty(tuple(out_shape), dtype=np.int32)
_gradient_2d(moving, moving_world2grid, moving_spacing, 
             static_grid2world, out, inside)

np.save('sl_gradient_2d.npy', out)

out = 255 * (out - out.min()) / (out.max() - out.min())

import matplotlib.pyplot as plt
from xvfbwrapper import Xvfb
with Xvfb() as xvfb:
	fig, ax = plt.subplots(1, 2)
	ax[0] = imshow(out[:, :, 0])
	ax[0] = set_title('grad_x')
	ax[1] = imshow(out[:, :, 1])
	ax[1] = set_title('grad_y')
	fig.savefig('sl_gradient_2d.png')
