# Symmetric Diffeomorphic Registration in 3D
#===========================================

# register two 3D volumes using SyN with Cross Correlation (CC) metric
import numpy as np
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
import os.path
from dipy.viz import regtools

# fetch 2 b0 volums, first b0 from Stanford HARDI dataset
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
stanford_b0 = np.squeeze(nib_stanford.get_data())[..., 0]

# fetch second b0 volume
from dipy.data.fetcher import fetch_syn_data, read_syn_data
fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
syn_b0 = np.array(nib_syn_b0.get_data())

# remove the skull from the b0's
from dipy.segment.mask import median_otsu
stanford_b0_masked, stanford_b0_mask = median_otsu(stanford_b0, 4, 4)
syn_b0_masked, syn_b0_mask = median_otsu(syn_b0, 4, 4)

static = stanford_b0_masked
static_affine = nib_stanford.affine
moving = syn_b0_masked
moving_affine = nib_syn_b0.affine

# already done a linear registration to roughly align two images
pre_align = np.array([[1.02783543e+00, -4.83019053e-02, -6.07735639e-02, -2.57654118e+00],
                      [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e+01],
                      [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e+01],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# resample the moving image on the static grid
from dipy.align.imaffine import AffineMap
affine_map = AffineMap(pre_align,
                       static.shape, static_affine,
                       moving.shape, moving_affine)
resampled = affine_map.transform(moving)
# overlay middle slices of the volumes
regtools.overlay_slices(static, resampled, None, 1, 
                        'Static', 'Moving', 'input_3d.png')

# use Cross Correlation metric
metric = CCMetric(3)

# set up multi-resolution strategy
level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

# execute the optimization, returns a DiffeomorphicMap object
mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align)
# warped the moving image
warped_moving = mapping.transform(moving)
regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving.png')
# inverse warped the static image
warped_static = mapping.transform_inverse(static)
regtools.overlay_slices(warped_static, moving, None, 1, 'Warped static', 'Moving', 'warped_static.png')
