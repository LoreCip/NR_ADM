import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import sys
sys.path.append("../")
from EFE_TR import d_r, d2_r


def f(x):
    return x**9

def fp(x):
    return 9*x**8

def fs(x):
    return 9*8*x**7

X, dx = np.linspace(-1, 1, 10000, retstep = True)
F  = f(X)
Fp = fp(X)
Fs = fs(X)

def test_d_r():
    d_x = d_r(F, dx)
    assert_allclose( Fp, d_x, atol=1e-7 )
    
def test_d2_r():
    d2_x = d2_r(F, dx)
    assert_allclose( Fs, d2_x, atol=1e-7 )
    
if __name__ == '__main__':
    test_d_r()
    print('Test 1/2 passed.')
    test_d2_r()
    print('Test 2/2 passed.')