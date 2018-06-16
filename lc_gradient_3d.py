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

np.save('lc_gradient_3d.npy', out)

out = np.asarray(out, dtype=np.float64)
out = 255 * (out - out.min()) / (out.max() - out.min())

slice_indices = np.array(out.shape) // 2
axial = np.asarray(out[:, :, slice_indices[2]]).astype(np.uint8)
coronal = np.asarray(out[:, slice_indices[1], :]).astype(np.uint8)
sagittal = np.asarray(out[slice_indices[0], :, :]).astype(np.uint8)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)
ax[0].imshow(axial)
ax[0].set_title('axial')
ax[1].imshow(coronal)
ax[1].set_title('coronal')
ax[2].imshow(sagittal)
ax[2].set_title('sagittal')
fig.savefig('lc_gradient_3d.png')
