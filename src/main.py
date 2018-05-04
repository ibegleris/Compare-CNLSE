import numpy as np
from ploting import Plot2d
from formulate import formulate_BNLSE, formulate_GNLSE
from combined_functions import overlap, dF_sidebands, sim_parameters, \
    float_range, pulse_propagation_adaptive, pulse_propagation_constant,\
    pulse_propagation_constant_timer
from scipy.constants import pi
import os


def pulse_GNLSE(P, sim_wind, int_fwm):
    pulses = []
    for i, p in zip(range(-1, 2), P):
        pulses.append(p**0.5 * np.exp(-0.5*(sim_wind.t_band / int_fwm.T0)**2) *
                      np.exp(1j * i * 2*pi * sim_wind.F * sim_wind.t))
    pulse = np.sum(pulses, axis=0)
    return pulse


def pulse_BNLSE(P, sim_wind, int_fwm):
    pulses = [p ** 0.5 * np.exp(-0.5 * (t_band / int_fwm.T0)**2)
              for p, t_band in zip(P, sim_wind.t_band)]
    return np.array(pulses)


def create_filestructure():
    os.system('mkdir output; mkdir output/data')


def inputs(mult_phi, dz=None, point_over=0, fr=None):
    "---------------constants---------------"
    time_me = False
    ss = 1
    n2 = 2.5e-20
    gama = 10e-3
    lamda_c = 1051.85e-9
    dz_less = 1000
    betas = np.array([0, 0, 0, 6.756e-2,    # propagation constants [ps^n/m]
                      -1.002e-4, 3.671e-7]) * 1e-3
    maxerr = 1e-12
    if fr is None:
        fr = 0.18
    "----------Combined parameters-----------"
    alphadB = 0

    P_p = 10
    P_s = 0
    P_i = 100e-3
    T0 = 50
    z = 14
    print('distance to propagate {} m'.format(z))
    df = 0.002  # frequency step in [Thz]
    lamp = lamda_c - 5e-9  # 0.5e-9
    if dz is not(None):
        z = float_range(0, z, dz)
    "----------------------------------------"
    "------------BNLSE parameters------------"
    N_b = 12+point_over
    Df_band = df * 2**N_b

    "------------GNLSE parameters------------"
    N_g = 18+point_over
    "----------------------------------------"

    return ss, n2, gama, lamda_c, dz_less, betas, maxerr, fr, alphadB, \
        z, df, lamp, P_p, P_s, P_i, T0, N_b, Df_band, N_g, time_me


def main(i_num, mult_phi, dz, point_over, fr):

    ss, n2, gama, lamda_c, dz_less, betas, maxerr, fr, alphadB, \
        z, df, lamp, P_p, P_s, P_i, T0, N_b, Df_band, N_g, time_me = inputs(
            mult_phi, dz=dz, point_over=point_over, fr=fr)

    M = overlap(n2, lamda_c, gama)

    F, fp = dF_sidebands(betas, lamp, lamda_c, n2, M, P_p, F_over=0, DF_over=0)

    int_fwm = sim_parameters(n2, 1, alphadB, betas, M, fr, T0)
    int_fwm.general_options(maxerr, ss)
    '-------------------------Formulate---------------------------'

    if type(z) is int:
        pulse_propagation = pulse_propagation_adaptive
        z_end = z
    else:
        if time_me:
            pulse_propagation = pulse_propagation_constant_timer
        else:
            pulse_propagation = pulse_propagation_constant
        z_end = z.end

    u_g, U_g, int_fwm_g, sim_wind_g, Dop_g, non_integrand_g = \
        formulate_GNLSE(int_fwm, N_g, z_end, dz_less, fp, df,
                        F, lamp, lamda_c, P_p, P_s, P_i, pulse_GNLSE)

    u_b, U_b, int_fwm_b, sim_wind_b, Dop_b, non_integrand_b = \
        formulate_BNLSE(int_fwm, N_b, z_end, dz_less, fp, df, F,
                        Df_band, lamp, lamda_c, P_p, P_s, P_i, pulse_BNLSE,
                        sim_wind_g.fv, sim_wind_g)
    print(sim_wind_b.df*1000)
    plotb_BNLSE = Plot2d(U_b, sim_wind_b, 'BNLSE',
                         extra_name=str(int(i_num)), Deltaf=0.4)
    plotb_BNLSE.plot_f(U_b, '0')
    plotb_BNLSE.plot_t(u_b, '0')
    plotb_BNLSE.save_data('0', u_b, U_b)

    plotg_GNLSE = Plot2d(U_g, sim_wind_g, 'GNLSE',
                         extra_name=str(int(i_num)), Deltaf=0.4)
    plotg_GNLSE.plot_f(U_g, '0')
    plotg_GNLSE.plot_t(u_g, '0')
    plotg_GNLSE.save_data('0', u_g, U_g)
    '------------------------BNLSE------------------------------------------'

    u_out_b, U_out_b, u_out_b_large, U_out_b_large, z_photo_vec =\
                pulse_propagation(u_b, U_b, int_fwm_b, sim_wind_b,
                              Dop_b, non_integrand_b.dAdzmm, z)
    plotb_BNLSE.plot_f(U_out_b, '1')
    plotb_BNLSE.plot_t(u_out_b, '1')
    plotb_BNLSE.save_data('1', u_out_b, U_out_b, u_out_b_large,
                          U_out_b_large, z_photo_vec, sim_wind_b.gama,
                          sim_wind_b.woffset)

    '--------------------------GNLSE----------------------'
    u_out_g, U_out_g, u_out_g_large, U_out_g_large, z_photo_vec = \
    pulse_propagation(u_g, U_g, int_fwm_g, sim_wind_g,
                      Dop_g, non_integrand_g.dAdzmm, z)
    plotg_GNLSE.plot_f(U_out_g, '1')
    plotg_GNLSE.plot_t(u_out_g, '1')
    plotg_GNLSE.save_data('1', np.array([0.]), np.array([0.]), np.array(
        [0.]), U_out_g_large, z_photo_vec, sim_wind_b.gama, sim_wind_g.woffset)
    print('\a')
    return None


if __name__ == '__main__':
    dZ = [10]
    for i in range(1, 15):
        dZ.append(dZ[i - 1]/2)
    dZ = [0.001]
    for mult_phi in (0.5,):
        for dz in dZ:
            for i_num, fr in enumerate((0.18, 0)):
                main(i_num, mult_phi, dz, 0, fr)
