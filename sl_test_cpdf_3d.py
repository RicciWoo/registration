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

nbins = 32

static = ((static.astype(np.float64) - static.min()) /
          (static.max() - static.min()))
moving = ((moving.astype(np.float64) - moving.min()) /
          (moving.max() - moving.min()))

static = np.array(static).astype(np.float64)
moving = np.array(moving).astype(np.float64)

smask = np.ones_like(static)
mmask = np.ones_like(moving)

smin = np.min(static[smask != 0])
smax = np.max(static[smask != 0])
mmin = np.min(moving[mmask != 0])
mmax = np.max(moving[mmask != 0])

padding = 2

sdelta = (smax - smin) / (nbins - 2 * padding)
mdelta = (mmax - mmin) / (nbins - 2 * padding)
smin = smin / sdelta - padding
mmin = mmin / mdelta - padding

joint = np.zeros(shape=(nbins, nbins))
smarginal = np.zeros(shape=(nbins,), dtype=np.float64)
mmarginal = np.zeros(shape=(nbins,), dtype=np.float64)

from dipy.align.parzenhist import _compute_pdfs_dense_3d
_compute_pdfs_dense_3d(static, moving, smask, mmask, smin,
                       sdelta, mmin, mdelta,
                       nbins, padding, joint,
                       smarginal, mmarginal)
