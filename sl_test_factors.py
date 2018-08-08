import numpy as np
from numpy.testing import assert_array_almost_equal
from dipy.align import floating
from dipy.align import crosscorr as cc

a = np.array(range(20*20), dtype=floating).reshape(20, 20)
b = np.array(range(20*20)[::-1], dtype=floating).reshape(20, 20)
a /= a.max()
b /= b.max()
for radius in [0, 1, 3, 6]:
    factors = np.asarray(cc.precompute_cc_factors_2d(a, b, radius))
    expected = np.asarray(cc.precompute_cc_factors_2d_test(a, b, radius))
    assert_array_almost_equal(factors, expected)