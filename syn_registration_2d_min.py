# Symmetric Diffeomorphic Registration in 2D
#===========================================

import numpy as np
from dipy.data import get_data
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import dipy.align.imwarp as imwarp
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

# register couple of slices from b0 iamge using Cross Correlation metric
# draw the current warped iamges after finishing each resolution
def callback_CC(sdr, status):
    #Status indicates at which stage of the optimization we currently are
    #For now, we will only react at the end of each resolution of the scale
    #space
    if status == imwarp.RegistrationStages.SCALE_END:
        #get the current images from the metric
        wmoving = sdr.metric.moving_image
        wstatic = sdr.metric.static_image
        #draw the images on top of each other with different colors
        regtools.overlay_images(wmoving, wstatic, 'Warped moving', 'Overlay', 'Warped static')

# load the data
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.segment.mask import median_otsu
fetch_syn_data()
t1, b0 = read_syn_data()
data = np.array(b0.get_data(), dtype = np.float64)

# remove the skull from the b0 volume
b0_mask, mask = median_otsu(data, 4, 4)

# select 2 slices to try the 2D registration
static = b0_mask[:, :, 40]
moving = b0_mask[:, :, 38]

# instantiate the Cross Correlation (CC) metric
sigma_diff = 3.0 # standard deviation of Gaussian kernel
radius = 4       # radius of window for evaluating local normalized CC
metric = CCMetric(2, sigma_diff, radius) # 2 is dimension of input images

# setup 3 levels for multi-resolution approach
level_iters = [100, 50, 25]
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
sdr.callback = callback_CC

# execute the optimization
mapping = sdr.optimize(static, moving)
warped = mapping.transform(moving)
# overlay input images
regtools.overlay_images(static, moving, 'Static', 'Overlay', 'Moving',
               't1_slices_input.png')
# transformed moving image
regtools.overlay_images(static, warped, 'Static', 'Overlay', 'Warped moving',
               't1_slices_res.png')
# inverse transformed static image
inv_warped = mapping.transform_inverse(static)
regtools.overlay_images(inv_warped, moving, 'Warped static', 'Overlay', 'moving',
               't1_slices_res2.png')
# plot the deformation
regtools.plot_2d_diffeomorphic_map(mapping, 5, 'diffeomorphic_map_b0s.png')
