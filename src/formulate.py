from combined_functions import *
import GNLSE_specific as GN
import BNLSE_specific as BN
from copy import deepcopy
import numpy as np
from scipy.constants import c, pi
from integrands import Integrand_BNLSE, Integrand_GNLSE
from combined_functions import dispersion_operator
from numpy.fft import fftshift
from scipy.fftpack import fft


def formulate_BNLSE(int_fwm, N_b, z, dz_less, fp, df, F, Df_band, lamp,
                    lamda_c, P_p, P_s, P_i, pulse, fv_g, sim_wind_g):
    int_fwm_b = deepcopy(int_fwm)
    int_fwm_b.propagation_parameters(N_b, z, dz_less)

    fv_b, where_g, f_centrals, band_grid_pos = \
                BN.fv_creator(fp, F, int_fwm_b, df, fv_g)
    sim_wind_b = BN.sim_window(
        fv_b, lamp, f_centrals, lamda_c, int_fwm_b, where_g)
    raman = BN.Raman_banded(sim_wind_b, sim_wind_g)
    sim_wind_b.hf = raman.set_raman_band()

    loss = BN.Loss(int_fwm_b, sim_wind_b, amax=0)
    int_fwm_b.alpha = loss.atten_func_full(fv_b)

    int_fwm_b.gama = np.array(
        [-1j * int_fwm.n2 * 2 * int_fwm_b.M * pi * (1e12 * f_c) / (c)
                for f_c in f_centrals])
    if int_fwm.ss == 0:
        int_fwm_b.gama = -1j * int_fwm.n2 * 2 * \
            int_fwm_b.M * pi * (1e12 * f_centrals[1]) / (c)
    sim_wind_b.gama = int_fwm_b.gama
    Dop = dispersion_operator(int_fwm.betas, lamda_c, int_fwm_b, sim_wind_b)

    noise_obj = BN.Noise(int_fwm_b, sim_wind_b)

    non_integrand = Integrand_BNLSE(int_fwm_b, sim_wind_b)
    P = [P_s, P_p, P_i]
    u = np.zeros([3, int_fwm_b.nt], dtype=np.complex128)
    u += pulse(P, sim_wind_b, int_fwm_b)
    u += noise_obj.noise_func(int_fwm_b)
    U = fftshift(fft(u), axes=-1)

    return u, U, int_fwm_b, sim_wind_b, Dop, non_integrand


def formulate_GNLSE(int_fwm, N_g, z, dz_less, fp, df, F, lamp,
                    lamda_c, P_p, P_s,P_i, pulse):
    int_fwm_g = deepcopy(int_fwm)
    int_fwm_g.propagation_parameters(N_g, z, dz_less)
    fv_g, where_g, f_centrals = GN.fv_creator(fp, df, F, int_fwm_g)
    sim_wind_g = GN.sim_window(fv_g, lamp, F, lamda_c, int_fwm_g, where_g)
    loss = GN.Loss(int_fwm_g, sim_wind_g, amax=0)
    int_fwm_g.gama = -1j * int_fwm.n2 * 2 * \
            int_fwm.M * pi * (1e12 * f_centrals[1]) / (c)
    int_fwm_g.alpha = loss.atten_func_full(fv_g)
    Dop = dispersion_operator(int_fwm_g.betas, lamda_c, int_fwm_g, sim_wind_g)

    noise_obj = GN.Noise(int_fwm_g, sim_wind_g)
    raman = GN.raman_object('load')
    raman.raman_load(sim_wind_g.t, sim_wind_g.dt)
    sim_wind_g.hf = raman.hf
    sim_wind_g.ht = raman.ht
    non_integrand = Integrand_GNLSE(int_fwm_g, sim_wind_g)
    P = [P_s, P_p, P_i]
    u = np.zeros([int_fwm_g.nt], dtype=np.complex128)
    u += pulse(P, sim_wind_g, int_fwm_g) + \
        noise_obj.noise_func(int_fwm_g)
    U = fftshift(fft(u), axes=-1)
    return u, U, int_fwm_g, sim_wind_g, Dop, non_integrand
