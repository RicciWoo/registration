
import numpy as np
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
# from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
# import os.path
from dipy.viz import regtools

from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
nib_stanford, _ = read_stanford_hardi()
# stanford_b0 = np.squeeze(nib_stanford.get_data())[..., 0]

from dipy.data.fetcher import fetch_syn_data, read_syn_data
fetch_syn_data()
_, nib_syn_b0 = read_syn_data()
# syn_b0 = np.array(nib_syn_b0.get_data())

static = np.load('stf_b0_masked.npy')
static_affine = nib_stanford.affine
moving = np.load('syn_b0_masked.npy')
moving_affine = nib_syn_b0.affine

pre_align = np.array([[1.02783543e+00, -4.83019053e-02, -6.07735639e-02, -2.57654118e+00],
                      [4.34051706e-03, 9.41918267e-01, -2.66525861e-01, 3.23579799e+01],
                      [5.34288908e-02, 2.90262026e-01, 9.80820307e-01, -1.46216651e+01],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

from dipy.align.imaffine import AffineMap
affine_map = AffineMap(pre_align,
                       static.shape, static_affine,
                       moving.shape, moving_affine)

resampled = affine_map.transform(moving)
regtools.overlay_slices(static, resampled, None, 1, 'Static', 'Moving', 'input_3d.png')

metric = CCMetric(3)

level_iters = [10, 10, 5]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

mapping = sdr.optimize(static, moving, static_affine, moving_affine, pre_align)

warped_moving = mapping.transform(moving)
regtools.overlay_slices(static, warped_moving, None, 1, 'Static', 'Warped moving', 'warped_moving.png')

warped_static = mapping.transform_inverse(static)
regtools.overlay_slices(warped_static, moving, None, 1, 'Warped static', 'Moving', 'warped_static.png')
