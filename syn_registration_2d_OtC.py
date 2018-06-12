# Symmetric Diffeomorphic Registration in 2D
#===========================================

import numpy as np
from dipy.data import get_data
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
from dipy.viz import regtools

# perform Circle-To-C experiment for deffeomorphic registration
fname_moving = get_data('reg_o')
fname_static = get_data('reg_c')
moving = np.load(fname_moving)
static = np.load(fname_static)
regtools.overlay_images(static, moving, 
                        'Static', 'Overlay', 'Moving', 'input_images.png')

# use Sum of Squared Differences (SSD) as similarity metric
dim = static.ndim
metric = SSDMetric(dim) 

# configurate for multi-resolution approach
level_iters = [200, 100, 50, 25]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter = 50)

# execute oprimization, returns DiffeomorphicMap object
mapping = sdr.optimize(static, moving)
# plot the deformation map
regtools.plot_2d_diffeomorphic_map(mapping, 10, 'diffeomorphic_map.png')
# warp the moving image
warped_moving = mapping.transform(moving, 'linear')
regtools.overlay_images(static, warped_moving,
                        'Static','Overlay','Warped moving',
                        'direct_warp_result.png')
# inverse warp the static image
warped_static = mapping.transform_inverse(static, 'linear')
regtools.overlay_images(warped_static, moving,'Warped static','Overlay','Moving', 
   'inverse_warp_result.png')
