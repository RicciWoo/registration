import numpy as np
from dipy.align import floating
from dipy.align import vector_fields as vfu
"""
Creates two random displacement field that exactly map pixels from an input
image to an output image. The resulting displacements and their
composition, although operating in physical space, map the points exactly
(up to numerical precision).
"""
np.random.seed(8315759)
input_shape = (10, 10, 10)
tgt_sh = (10, 10, 10)
# create a simple affine transformation
ns = input_shape[0]
nr = input_shape[1]
nc = input_shape[2]
s = 1.5
t = 2.5
trans = np.array([[1, 0, 0, -t * ns],
                  [0, 1, 0, -t * nr],
                  [0, 0, 1, -t * nc],
                  [0, 0, 0, 1]])
trans_inv = np.linalg.inv(trans)
scale = np.array([[1 * s, 0, 0, 0],
                  [0, 1 * s, 0, 0],
                  [0, 0, 1 * s, 0],
                  [0, 0, 0, 1]])
gt_affine = trans_inv.dot(scale.dot(trans))

# create two random displacement fields
input_grid2world = gt_affine
target_grid2world = gt_affine

disp1, _ = vfu.create_random_displacement_3d(
    np.array(input_shape, dtype=np.int32),
    input_grid2world, np.array(tgt_sh, dtype=np.int32),
    target_grid2world)
disp1 = np.array(disp1, dtype=floating)

disp2, _ = vfu.create_random_displacement_3d(
    np.array(input_shape, dtype=np.int32), input_grid2world,
    np.array(tgt_sh, dtype=np.int32), target_grid2world)
disp2 = np.array(disp2, dtype=floating)

# compose the displacement fields
target_world2grid = np.linalg.inv(target_grid2world)
premult_index = target_world2grid.dot(input_grid2world)
premult_disp = target_world2grid

time_scaling = 1.0

import line_profiler as lp
profile = lp.LineProfiler(vfu.compose_vector_fields_3d)
profile.runcall(vfu.compose_vector_fields_3d, disp1,
                disp2 / time_scaling, premult_index,
                premult_disp, time_scaling, None)
profile.print_stats()
