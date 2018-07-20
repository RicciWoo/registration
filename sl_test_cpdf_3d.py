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

static = ((static.astype(np.float64) - static.min()) /
          (static.max() - static.min()))
moving = ((moving.astype(np.float64) - moving.min()) /
          (moving.max() - moving.min()))

static = np.array(static).astype(np.float64)
moving = np.array(moving).astype(np.float64)

from dipy.align.imaffine import AffineMap
dim = len(static.shape)
starting_affine = np.eye(dim + 1)
affine_map = AffineMap(starting_affine, static.shape, static_grid2world,
                       moving.shape, moving_grid2world)

static_values = static
moving_values = affine_map.transform(moving)

from dipy.align.parzenhist import ParzenJointHistogram
nbins = 32
histogram = ParzenJointHistogram(nbins)
histogram.update_pdfs_dense(static_values, moving_values)

np.save('sl_aff_par_cpdf_joint', histogram.joint)
np.save('sl_aff_par_cpdf_smarg', histogram.smarginal)
np.save('sl_aff_par_cpdf_mmarg', histogram.mmarginal)
