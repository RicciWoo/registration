import numpy as np
# from numpy.testing import assert_array_almost_equal
from dipy.align import floating
from dipy.align import crosscorr as cc

a = np.array(range(20*20), dtype=floating).reshape(20, 20)
b = np.array(range(20*20)[::-1], dtype=floating).reshape(20, 20)
a /= a.max()
b /= b.max()
radius = 6
factors = np.asarray(cc.precompute_cc_factors_2d(a, b, radius))
np.save('sl_syn_factors.npy', factors)
# expected = np.asarray(cc.precompute_cc_factors_2d_test(a, b, radius))
# assert_array_almost_equal(factors, expected)
