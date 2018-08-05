
import numpy as np
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
# from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
# import os.path
from dipy.viz import regtools
from xvfbwrapper import Xvfb

from dipy.data import fetch_stanford_hardi, read_stanford_hardi
fetch_stanford_hardi()
nib_stanford, _ = read_stanford_hardi()
stanford_b0 = np.squeeze(nib_stanford.get_data())[..., 0]

from dipy.data.fetcher import fetch_syn_data, read_syn_data
fetch_syn_data()
_, nib_syn_b0 = read_syn_data()
syn_b0 = np.array(nib_syn_b0.get_data())

from dipy.segment.mask import median_otsu
stanford_b0_masked, _ = median_otsu(stanford_b0, 4, 4)
syn_b0_masked, _ = median_otsu(syn_b0, 4, 4)

np.save('stf_b0_masked.npy', stanford_b0_masked)
np.save('syn_b0_masked.npy', syn_b0_masked)
