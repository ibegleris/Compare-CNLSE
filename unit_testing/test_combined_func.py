import sys
sys.path.append('src')
from numpy.testing import assert_allclose, assert_raises
from combined_functions import *
import pytest
import numpy as np
from test_BNLSE import specific_variables
from BNLSE_specific import Loss as Lossb

def test_dbm2w():
	assert dbm2w(30) == 1


def test1_w2dbm():
	assert w2dbm(1) == 30


def test2_w2dbm():
	a = np.zeros(100)
	floor = np.random.rand(1)[0]
	assert_allclose(w2dbm(a,-floor), -floor*np.ones(len(a)))


def test3_w2dbm():
	with pytest.raises(ZeroDivisionError):
		w2dbm(-1)

def test_my_arange():
	assert_allclose(np.arange(0,1,0.1), my_arange(0,1,0.1)[:-1])


def test_overlap():
	n2 = 2.5e-20
	lamda_g = 1051.85e-9
	gama = 10e-3
	assert_allclose(overlap(n2, lamda_g, gama),66962850756.48404)


def test_dF_sidebands():
    """
	Tests of the ability of dF_sidebands to find the
	sidebands expected for predetermined conditions.
    """
    lamp = 1048.17e-9
    lamda0  = 1051.85e-9
    betas =  0, 0,0, 6.756e-2 *1e-3, -1.002e-4 * 1e-3, 3.671*1e-7 * 1e-3
    n2 = 2.5e-20
    gama = 10e-3
    M = overlap(n2, lamda0, gama)
    F, f_p = dF_sidebands(betas, lamp,lamda0, n2, M,5)
  

    f_s, f_i = f_p - F, f_p + F
    lams, lami = (1e-3*c/i for i in (f_s, f_i))
    assert lams, lami == (1200.2167948665879, 930.31510086250455)


def test_dispersion():
    M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N, lamda_c, lamda,  \
    betas, fv,  f_centrals = specific_variables()
    int_fwm.alphadB = 0.1
    loss = Lossb(int_fwm, sim_wind, amax = 0.1)
    alpha_func = loss.atten_func_full(sim_wind.fv)
    int_fwm.alphadB = alpha_func
    int_fwm.alpha = int_fwm.alphadB


    betas_disp = dispersion_operator(betas,lamda_c,int_fwm,sim_wind)
    #np.savetxt('unit_testing/testing_data//exact_dispersion.txt',betas_disp.view(np.float))


    betas_exact = np.loadtxt('unit_testing/testing_data/exact_dispersion.txt').view(complex)
    assert_allclose(betas_disp,betas_exact)
