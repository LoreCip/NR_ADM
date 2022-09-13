import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import sys
sys.path.append("../")
from Constraint import comp_Hconstraint
from FieldsClass import Fields

N = 40000 + 2

r = np.array([(j - 0.5)*(2.5/(N-2)) for j in range(1,N-1)], dtype = np.float64)
psi = 1 + 1 / 4 / r

r = np.insert(r, 0, [None, None])
psi = np.insert(psi, 0, [None, None])


true_IC = np.zeros(8*N)
true_IC[0:2*N] = 1


def test_Geodesic():
    true_IC[6*N:7*N] = 1 
    
    fields = Fields(N = N - 2)
    fields.IC_GeodesicSlicing()
    
    assert_allclose(true_IC, fields.fields, atol = 1e-7)
    
def test_Hcon_Geo():
    fields = Fields(N = N - 2)
    fields.IC_GeodesicSlicing()
    Hc = comp_Hconstraint(fields)
    
    assert_almost_equal(Hc, 0, decimal = 2) 
    
def test_OnePlusLog():
    true_IC[6*N:7*N] = (1 - 1 / 4 / r) / psi   
    
    fields = Fields(N = N - 2)
    fields.IC_1plusLogSlicing()

    assert_allclose(true_IC, fields.fields, atol = 1e-7)
    
def test_Hcon_Log():    
    fields = Fields(N = N - 2)
    fields.IC_1plusLogSlicing()
    Hc = comp_Hconstraint(fields)
    
    assert_almost_equal(Hc, 0, decimal = 2) 
    

if __name__ == '__main__':
    test_Geodesic()
    print('Test 1/4 passed.')
    test_OnePlusLog()
    print('Test 2/4 passed.')
    test_Hcon_Log()
    print('Test 3/4 passed.')
    test_Hcon_Geo()
    print('Test 4/4 passed.')