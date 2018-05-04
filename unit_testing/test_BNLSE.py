import sys
sys.path.append('src')
from combined_functions import dF_sidebands, sim_parameters,\
     overlap, pulse_propagation_adaptive, pulse_propagation_constant,\
     float_range
from BNLSE_specific import *
import GNLSE_specific as GN
from copy import deepcopy
from numpy.testing import assert_allclose, assert_raises
from test_GNLSE import specific_variables
from formulate import formulate_BNLSE, formulate_GNLSE
from test_GNLSE import pulse as pulse_GNLSE



def specific_variables():
    n2 = 2.5e-20
    alphadB = 0
    maxerr = 1e-13
    ss = 1
    gama = 10e-3
    lamda_c = 1051.85e-9
    lamp = 1046.85e-9
    betas = np.array([0, 0, 0, 6.756e-2,
                      -1.002e-4, 3.671e-7])*1e-3
    T0 = 10
    P_p = 5
    P_s = 1
    P_i = 2
    fr = 0.
    fwhm = 1
    N_b = 10
    N_g = 17
    z = 18
    dz_less = 100
    df = 0.010
    M = overlap(n2, lamda_c, gama)
    F, fp = dF_sidebands(betas, lamp, lamda_c, n2, M, P_p)

    int_fwm = sim_parameters(n2, 1, alphadB, betas, M, fr, T0)
    int_fwm.general_options(maxerr, ss)


    int_fwm_g = deepcopy(int_fwm)
    int_fwm_g.propagation_parameters(N_g, z, dz_less)   
    #print(int_fwm.dz)
    #sys.exit()
    #f_centrals = [fp + i * F for i in range(-1,2)]

    fv_g, where_g,f_centrals = GN.fv_creator(fp, df, F, int_fwm_g)

    int_fwm = deepcopy(int_fwm)
    int_fwm.propagation_parameters(N_b, z, dz_less)    
    fv,where,f_centrals,band_grid_pos  = fv_creator(fp, F, int_fwm,df, fv_g)
    #sim_wind = sim_window(fv, lamp, f_centrals, lamda_c, int_fwm_b, where_g)
    sim_wind = sim_window(fv, lamp, f_centrals, lamda_c, int_fwm, where_g)
    return M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N_b, lamda_c, lamp, \
        betas, fv,  f_centrals



M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N, lamda_c, lamda,  \
    betas, fv,  f_centrals = specific_variables()


#fv, where_g = fv_creator(f_centrals[1], fv[1] - fv[0], f_centrals[1] - f_centrals[0], int_fwm, fv)
#sim_wind = sim_window(fv, lamp,F, lamda_c, int_fwm, where_g)
def test_dF_sidebands():
    """
        Tests of the ability of dF_sidebands to find the
        sidebands expected for predetermined conditions.
    """
    lamp = 1048.17e-9
    lamda0 = 1051.85e-9
    betas = 0, 0, 0, 6.756e-2 * 1e-3, -1.002e-4 * 1e-3, 3.671*1e-7 * 1e-3

    F, f_p = dF_sidebands(betas, lamp, lamda0, n2, M, 5)

    f_s, f_i = f_p - F, f_p + F
    lams, lami = (1e-3*c/i for i in (f_s, f_i))
    assert lams, lami == (1200.2167948665879, 930.31510086250455)


def test_noise():
    noise = Noise(int_fwm, sim_wind)
    n1 = noise.noise_func(int_fwm)
    n2 = noise.noise_func(int_fwm)
    print(n1, n2)
    assert_raises(AssertionError, assert_allclose, n1, n2)


class Test_loss:
    def test_loss1(a):
        loss = Loss(int_fwm, sim_wind, amax=alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        assert_allclose(alpha_func, np.ones_like(alpha_func)*alphadB/4.343)

    def test_loss2(a):

        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        maxim = np.max(alpha_func)
        assert_allclose(maxim, 2*alphadB/4.343)

    def test_loss3(a):
        loss = Loss(int_fwm, sim_wind, amax=2*alphadB)
        alpha_func = loss.atten_func_full(sim_wind.fv)
        minim = np.min(alpha_func)
        assert minim == alphadB/4.343

class Test_energy_conserve_adaptive():

    "---------------constants---------------"
    ss = 1
    n2 = 2.5e-20
    gama = 10e-3
    lamda_c = 1051.85e-9
    dz_less = 100
    betas = np.array([0, 0, 0, 6.756e-2,    # propagation constants [ps^n/m]
                      -1.002e-4, 3.671e-7]) * 1e-3
    maxerr = 1e-13
    fr = 0.18
    "----------Combined parameters-----------"
    alphadB = 0
    z = 18
    df = 0.01  # frequency step in [Thz]
    lamp = 1048e-9
    P_p = 20
    P_s = 1
    P_i = 1
    T0 = 10  # [ps]
    "----------------------------------------"
    "------------BNLSE parameters------------"
    N_b = 10
    Df_band = df * 2**N_b
    "------------GNLSE parameters------------"
    N_g = 14
    "----------------------------------------"

    M = overlap(n2, lamda_c, gama)
    F, fp = dF_sidebands(betas, lamp, lamda_c, n2, M, P_p)



    sim_wind_g = \
            formulate_GNLSE(int_fwm, N_g, z, dz_less, fp, df,
                        F, lamp, lamda_c, P_p, P_s,P_i, pulse_GNLSE)[3]

    def test_energy_conserve_s0_pulse(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, pulse, self.sim_wind_g.fv,
                            self.sim_wind_g)
       
        ss = 0
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1_pulse(self):

        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, pulse, self.sim_wind_g.fv,
                            self.sim_wind_g)
        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s0_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, cw, self.sim_wind_g.fv,
                            self.sim_wind_g)
        ss = 0
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, cw, self.sim_wind_g.fv,
                            self.sim_wind_g)
        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)


class Test_energy_conserve_constant():

    "---------------constants---------------"
    ss = 1
    n2 = 2.5e-20
    gama = 10e-3
    lamda_c = 1051.85e-9
    dz_less = 100
    betas = np.array([0, 0, 0, 6.756e-2,    # propagation constants [ps^n/m]
                      -1.002e-4, 3.671e-7]) * 1e-3
    maxerr = 1e-13
    fr = 0.18
    "----------Combined parameters-----------"
    alphadB = 0
    z = 1
    dz = 0.01
    z = float_range(0, z, dz)
    df = 0.01  # frequency step in [Thz]
    lamp = 1048e-9
    P_p = 20
    P_s = 1
    P_i = 1
    T0 = 10  # [ps]
    "----------------------------------------"
    "------------BNLSE parameters------------"
    N_b = 10
    Df_band = df * 2**N_b
    "------------GNLSE parameters------------"
    N_g = 14
    "----------------------------------------"

    M = overlap(n2, lamda_c, gama)
    F, fp = dF_sidebands(betas, lamp, lamda_c, n2, M, P_p)



    sim_wind_g = \
            formulate_GNLSE(int_fwm, N_g, z.end, dz_less, fp, df,
                        F, lamp, lamda_c, P_p, P_s,P_i, pulse_GNLSE)[3]

    def test_energy_conserve_s0_pulse(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, pulse, self.sim_wind_g.fv,
                            self.sim_wind_g)
       
        ss = 0
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1_pulse(self):

        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, pulse, self.sim_wind_g.fv,
                            self.sim_wind_g)
        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s0_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, cw, self.sim_wind_g.fv,
                            self.sim_wind_g)
        ss = 0
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_BNLSE(int_fwm, self.N_b, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.Df_band, self.lamp,
                            self.lamda_c, self.P_p,self.P_s, self.P_i, cw, self.sim_wind_g.fv,
                            self.sim_wind_g)
        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.sum(np.linalg.norm(u_b, 2, axis=-1)**2)
        E2 = np.sum(np.linalg.norm(u_out_b, 2, axis=-1)**2)

        assert_allclose(E1, E2)


def pulse(P, sim_wind, int_fwm):
    return np.array([p ** 0.5 * np.exp(-0.5 * (t_band / int_fwm.T0)**2)\
             for p, t_band in zip(P, sim_wind.t_band)])


def cw(P, sim_wind, int_fwm):
    woff1 = (sim_wind.p_pos[1] + (int_fwm.nt) // 2) * 2 * pi * sim_wind.df[sim_wind.p_pos[0]]

    return np.array([p ** 0.5 * np.exp(1j * (woff1) * sim_wind.t[sim_wind.p_pos[0]])\
             for p, t_band in zip(P, sim_wind.t_band)])
