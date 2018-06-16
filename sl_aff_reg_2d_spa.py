# Affine Registration in 3D
#==========================

import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
                                   RigidTransform2D,
                                   AffineTransform2D)
from xvfbwrapper import Xvfb

# fetch static image from the Stanford HARDI dataset
fetch_stanford_hardi()
nib_stanford, gtab_stanford = read_stanford_hardi()
static = np.squeeze(nib_stanford.get_data())[..., 0]
static_grid2world = nib_stanford.affine

# fetch moving image
fetch_syn_data()
nib_syn_t1, nib_syn_b0 = read_syn_data()
moving = np.array(nib_syn_b0.get_data())
moving_grid2world = nib_syn_b0.affine

static = static[:, :, 37]
moving = moving[:, :, 37]
static_grid2world = np.delete(static_grid2world, 2, 0)
static_grid2world = np.delete(static_grid2world, 2, 1)
moving_grid2world = np.delete(moving_grid2world, 2, 0)
moving_grid2world = np.delete(moving_grid2world, 2, 1)

# resample moving image on a grid of same dimemsions as static image
identity = np.eye(3)
affine_map = AffineMap(identity,
                       static.shape, static_grid2world,
                       moving.shape, moving_grid2world)
resampled = affine_map.transform(moving)
with Xvfb() as xvfb:
  regtools.overlay_images(static, resampled, "Static", 
                          "Overlay", "Moving", "resampled.png")

# align the centers of mass of two images
c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)
transformed = c_of_mass.transform(moving)
with Xvfb() as xvfb:
  regtools.overlay_images(static, resampled, "Static", 
                          "Overlay", "Moving", "transformed_com.png")

# create similarity metric (Mutual Information)
nbins = 32           # number of bins to discretize joint & marginal PDF
sampling_prop = 0.8  # percentage of voxels for computing PDFs, None: 100%
metric = MutualInformationMetric(nbins, sampling_prop)

# multi-resolution strategy, building Guassian Pyramid
level_iters = [10000, 1000, 100] # number of iterations
sigmas = [2.0, 1.0, 0.0]         # sd of Gaussian kernel
factors = [4, 2, 1]              # sub-sampling factors

# instantiate registration class
affreg = AffineRegistration(metric=metric,
                            level_iters=level_iters,
                            sigmas=sigmas,
                            factors=factors)

# First, look for an optimal translation
transform = TranslationTransform2D()
params0 = None
starting_affine = c_of_mass.affine
translation = affreg.optimize(static, moving, transform, params0,
                              static_grid2world, moving_grid2world,
                              starting_affine=starting_affine)
transformed = translation.transform(moving)
with Xvfb() as xvfb:
  regtools.overlay_images(static, transformed, "Static", 
                          "Overlay", "Moving", "transformed_trans.png")

# Then, refine with a rigid transform
transform = RigidTransform2D()
params0 = None
starting_affine = translation.affine
rigid = affreg.optimize(static, moving, transform, params0,
                        static_grid2world, moving_grid2world,
                        starting_affine=starting_affine)
transformed = rigid.transform(moving)
with Xvfb() as xvfb:
  regtools.overlay_images(static, transformed, "Static", 
                          "Overlay", "Moving", "transformed_rigid.png")

# Finally, refine with a full affine transorm
transform = AffineTransform2D()
params0 = None
starting_affine = rigid.affine
affine = affreg.optimize(static, moving, transform, params0,
                         static_grid2world, moving_grid2world,
                         starting_affine=starting_affine)
transformed = affine.transform(moving)
with Xvfb() as xvfb:
  regtools.overlay_images(static, transformed, "Static", 
                          "Overlay", "Moving", "transformed_affine.png")
