import sys
sys.path.append('src')
from combined_functions import dF_sidebands, sim_parameters, overlap
from GNLSE_specific import *
from copy import deepcopy
from numpy.testing import assert_allclose, assert_raises
from formulate import formulate_GNLSE
from scipy.integrate import simps
from combined_functions import pulse_propagation_adaptive, pulse_propagation_constant,\
float_range
def specific_variables():
    n2 = 2.5e-20
    alphadB = 0
    maxerr = 1e-13
    ss = 1
    gama = 10e-3
    lamda_c = 1051.85e-9
    lamp = 1048e-9
    lams = 1245.98
    betas = np.array([0, 0, 0, 6.756e-2,
                      -1.002e-4, 3.671e-7])*1e-3
    T0 = 10
    P_p = 5
    P_s = 1
    P_i = 2
    fr = 0.18
    fwhm = 1
    N_b = 10
    z = 18
    dz_less = 100
    df = 0.010
    M = overlap(n2, lamda_c, gama)
    F, fp = dF_sidebands(betas, lamp, lamda_c, n2, M, P_p)

    int_fwm = sim_parameters(n2, 1, alphadB, betas, M, fr, T0)
    int_fwm.general_options(maxerr, ss)
    int_fwm = deepcopy(int_fwm)
    int_fwm.propagation_parameters(N_b, z, dz_less)
    #print(int_fwm.dz)
    #sys.exit()
    f_centrals = [fp + i * F for i in range(-1,2)]
    fv, where_g, f_centrals = fv_creator(fp, df, F, int_fwm)
    #sim_wind = sim_window(fv, lamp, f_centrals, lamda_c, int_fwm_b, where_g)
    sim_wind = sim_window(fv, lamp,F, lamda_c, int_fwm, where_g)
    return M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N_b, lamda_c, lamp, lams, \
        betas, fv,  f_centrals


M, int_fwm, sim_wind, n2, alphadB, maxerr, ss, N, lamda_c, lamda, lams, \
    betas, fv,  f_centrals = specific_variables()


def test_noise():
    noise = Noise(int_fwm, sim_wind)
    n1 = noise.noise_func(int_fwm)
    n2 = noise.noise_func(int_fwm)
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



class Test_energy_conserve_split():

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
    P_p = 5
    P_s = 1
    P_i = 0
    T0 = 1  # [ps]
    "----------------------------------------"
    "------------BNLSE parameters------------"
    N_b = 10
    Df_band = df * 2**N_b
    "------------GNLSE parameters------------"
    N_g = 14
    "----------------------------------------"

    M = overlap(n2, lamda_c, gama)
    F, fp = dF_sidebands(betas, lamp, lamda_c, n2, M, P_p)

    int_fwm = sim_parameters(n2, 1, alphadB, betas, M, fr, T0)
    int_fwm.general_options(maxerr, ss)
    int_fwm = deepcopy(int_fwm)
    int_fwm.propagation_parameters(N_g, z, dz_less)

    def test_energy_conserve_s0_pulse(self):
        u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z, self.dz_less, self.fp, self.df,
                        self.F, self.lamp, self.lamda_c, self.P_p, self.P_s,self.P_i, pulse)
        ss = 0
        #u_out_b, U_out_b = self.pulse_propagation(
        #    u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g.dAdzmm, self.z)
        u_out_g, U_out_g, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g.dAdzmm, self.z)
        E1 = np.linalg.norm(u_g, 2, axis=-1)
        E2 = np.linalg.norm(u_out_g, 2, axis=-1)

        assert_allclose(E1, E2, atol = 1e-3)

    def test_energy_conserve_s1_pulse(self):

        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.lamp, self.lamda_c,
                            self.P_p,self.P_s, self.P_i, pulse)
        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.linalg.norm(u_b, 2, axis=-1)
        E2 = np.linalg.norm(u_out_b, 2, axis=-1)

        assert_allclose(E1, E2, atol = 1e-3)

    def test_energy_conserve_s0_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.lamp, self.lamda_c,
                            self.P_p,self.P_s, self.P_i, cw)
        ss = 0
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.linalg.norm(u_b, 2, axis=-1)
        E2 = np.linalg.norm(u_out_b, 2, axis=-1)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z, self.dz_less,
                            self.fp, self.df, self.F, self.lamp, self.lamda_c,
                            self.P_p,self.P_s, self.P_i, cw)

        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_adaptive(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm)
        E1 = np.linalg.norm(u_b, 2, axis=-1)
        E2 = np.linalg.norm(u_out_b, 2, axis=-1)

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


    int_fwm = sim_parameters(n2, 1, alphadB, betas, M, fr, T0)
    int_fwm.general_options(maxerr, ss)
    int_fwm = deepcopy(int_fwm)
    int_fwm.propagation_parameters(N_g, z.end, dz_less)

    def test_energy_conserve_s0_pulse(self):
        u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z.end, self.dz_less, self.fp, self.df,
                        self.F, self.lamp, self.lamda_c, self.P_p, self.P_s,self.P_i, pulse)
        ss = 0
        #u_out_b, U_out_b = self.pulse_propagation(
        #    u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g.dAdzmm, self.z)
        u_out_g, U_out_g, temp1,temp2,temp3 = pulse_propagation_constant(
            u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g.dAdzmm, self.z)
        E1 = np.linalg.norm(u_g, 2, axis=-1)
        E2 = np.linalg.norm(u_out_g, 2, axis=-1)

        assert_allclose(E1, E2, atol = 1e-2)

    def test_energy_conserve_s1_pulse(self):

        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.lamp, self.lamda_c,
                            self.P_p,self.P_s, self.P_i, pulse)
        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.linalg.norm(u_b, 2, axis=-1)
        E2 = np.linalg.norm(u_out_b, 2, axis=-1)

        assert_allclose(E1, E2, atol = 1e-2)

    def test_energy_conserve_s0_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.lamp, self.lamda_c,
                            self.P_p,self.P_s, self.P_i, cw)
        ss = 0
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.linalg.norm(u_b, 2, axis=-1)
        E2 = np.linalg.norm(u_out_b, 2, axis=-1)

        assert_allclose(E1, E2)

    def test_energy_conserve_s1_cw(self):
        u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
            formulate_GNLSE(self.int_fwm, self.N_g, self.z.end, self.dz_less,
                            self.fp, self.df, self.F, self.lamp, self.lamda_c,
                            self.P_p,self.P_s, self.P_i, cw)

        ss = 1
        u_out_b, U_out_b, temp1,temp2,temp3 = pulse_propagation_constant(
            u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b.dAdzmm, self.z)
        E1 = np.linalg.norm(u_b, 2, axis=-1)
        E2 = np.linalg.norm(u_out_b, 2, axis=-1)

        assert_allclose(E1, E2)




def pulse(P, sim_wind, int_fwm):
    pulses = []
    for i, p in zip(range(-1,2), P):
        pulses.append(p ** 0.5 * np.exp(-0.25*(sim_wind.t_band  / int_fwm.T0)**2)*\
                      np.exp(1j * i *2*pi* sim_wind.F *sim_wind.t))
    pulses = np.asarray(pulses)
    #for i,p in enumerate(pulses):
    #    norm = simps(np.abs(p)**2,sim_wind.t)
    #    if norm != 0:
    #        pulses[i, :] /= (0.5*norm)**0.5
    pulse = np.sum(pulses, axis = 0)
    print('total power is:',simps(np.abs(pulse)**2,sim_wind.t))
    return pulse


def cw(P, sim_wind, int_fwm):
    woff1 = (sim_wind.p_pos + (int_fwm.nt) // 2) * 2 * pi * sim_wind.df
    return P ** 0.5 * np.exp(1j * (woff1) * sim_wind.t[sim_wind.p_pos])


def cw(P, sim_wind, int_fwm):
    woff1 = (sim_wind.p_pos + (int_fwm.nt) // 2) * 2 * pi * sim_wind.df
    pulses = []
    for i, p in zip(range(-1,2), P):
        pulses.append(p ** 0.5 * np.exp(1j * (woff1) \
                        * sim_wind.t[sim_wind.p_pos]))
    pulses = np.asarray(pulses)
    #for i,p in enumerate(pulses):
    #    norm = simps(np.abs(p)**2,sim_wind.t)
    #    if norm != 0:
    #        pulses[i, :] /= (0.5*norm)**0.5
    #print(pulses.shape, sim_wind.t.shape)
    pulse = np.sum(pulses, axis = 0)
    #print('total power is:',simps(np.abs(pulse)**2,sim_wind.t))
    return pulse