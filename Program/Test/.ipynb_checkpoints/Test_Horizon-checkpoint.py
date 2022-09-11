import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

import sys
sys.path.append("../")
from Horizon import comp_appHorizon
from FieldsClass import Fields

N = 10000
fields_geo = Fields(N = N)
fields_geo.IC_GeodesicSlicing()
root_geo, surf_geo = comp_appHorizon(fields_geo)

fields_OPL = Fields(N = N)
fields_OPL.IC_1plusLogSlicing()
root_OPL, surf_OPL = comp_appHorizon(fields_OPL)
    
def test_horizon_Geo():
    assert_almost_equal(root_geo, 0.25)
    
def test_surface_Geo():    
    assert_almost_equal(surf_geo, 4 * np.pi * 16 * 0.25**2)

def test_horizon_OPL():
    assert_almost_equal(root_OPL, 0.25)
    
def test_surface_OPL():    
    assert_almost_equal(surf_OPL, 4 * np.pi * 16 * 0.25**2)



if __name__ == '__main__':
    test_horizon_Geo()
    print('Test 1/2 passed.')
    test_surface_Geo()
    print('Test 2/2 passed.')
    test_horizon_OPL()
    print('Test 3/4 passed.')
    test_surface_OPL()
    print('Test 4/4 passed.')