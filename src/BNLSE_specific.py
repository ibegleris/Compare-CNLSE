import numpy as np
from combined_functions import check_ft_grid
from scipy.constants import pi, c, Planck
from numpy.fft import fftshift
from time import time
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat
from scipy.interpolate import InterpolatedUnivariateSpline


def fv_creator(fp, F, int_fwm, df, fv_g):
    """
    Cretes 3 split frequency grid set up around the waves from degenerate
    FWM. The central freuency of the bands is determined by the non-depleted
    pump approximation and is power dependent. The wideness of these bands
    is determined inputed. This returns an array of shape [3, nt] with
    each collumn holding the data of the 3 frequency bands.
    """

    f_centrals = [fp + i * F for i in range(-1, 2)]
    band_grid_pos = [np.where(fc == fv_g)[0][0] for fc in f_centrals]
    fv = np.zeros([3, int_fwm.nt])
    df = fv_g[1] - fv_g[0]
    if (band_grid_pos[0] - int_fwm.nt//2 < 0) or\
            (band_grid_pos[2] + int_fwm.nt//2 > len(fv_g)):
        sys.exit('The banded is going off the GNLSE grid.')

    for i, b in enumerate(band_grid_pos):
        j, k = b - int_fwm.nt//2, b + int_fwm.nt//2
        fv[i, :] = fv_g[j:k]
    assert not(np.any(0 in fv))
    for f in fv:
        for ff in f:
            assert ff in fv_g

    p_pos = np.where(np.abs(fv - fp) == np.min(np.abs(fv - fp)))
    p_pos = [p_pos[0][0], p_pos[1][0]]
    where = [p_pos]
    check_ft_grid(fv, df)

    return fv, where, f_centrals, band_grid_pos


class sim_window(object):

    def __init__(self, fv, lamda, f_centrals, lamda_c, int_fwm, where):

        self.fv = fv
        self.lamda = lamda

        self.deltaf = np.array([np.max(f) - np.min(f) for f in fv])  # [THz]
        self.df = self.deltaf/int_fwm.nt  # [THz]
        self.T = 1 / self.df  # Time window (period)[ps]
        self.fmed = np.array([0.5*(f[-1] + f[0])*1e12 for f in fv])  # [Hz]

        self.type = 'BNLSE'
        self.woffset = 2*pi*(self.fmed[1] - c/lamda)*1e-12  # [rad/ps]

        # central angular frequency [rad/ps]
        self.w0 = 2*pi*np.asarray(f_centrals)

        self.tsh = 1/self.w0  # shock time [ps]
        self.dt = self.T/int_fwm.nt  # timestep (dt)     [ps]
        self.t = np.array(
            [(range(int_fwm.nt) - np.ones(int_fwm.nt) * int_fwm.nt/2) *
                dt for dt in self.dt])
        self.w = np.array(
            [fftshift(2 * pi * (fv - 1e-12*self.fmed[1]), axes=-1)
                for fv in self.fv])

        self.w_bands = np.array([fftshift(2 * pi * (fv - fc),
                                axes=-1) for fv, fc
                                in zip(self.fv, f_centrals)])
        self.t_band = self.t  # [1,:]
        self.lv = 1e-3*c/self.fv

        self.zv = int_fwm.dzstep*np.asarray(range(0, 2))
        # self.w_tiled = fftshift(
        #    2*pi * (self.fv - 1e-12*c/lamda_g),
        #         axes = -1)
        self.w_tiled = np.copy(self.w_bands)
        self.p_pos = where[0]
        self.Omega = 2 * pi * np.abs(f_centrals[1] - f_centrals[0])
        self.f_centrals = f_centrals
        self.F = f_centrals[1] - f_centrals[0]


class Loss(object):

    def __init__(self, int_fwm, sim_wind, amax=None, apart_div=16):
        """
        Initialise the calss Loss, takes in the general parameters and
        the freequenbcy window. From that it determines where the loss
        will become freequency dependent. With the default value being an 16th
        of the difference of max and min.

        """
        self.alpha = int_fwm.alphadB/4.343
        if amax is None:
            self.amax = self.alpha
        else:
            self.amax = amax/4.343

        self.flims_large = [(np.min(f), np.max(f)) for f in sim_wind.fv]

        self.apart = [np.abs(films[1] - films[0])
                      for films in self.flims_large]
        self.apart = [i/apart_div for i in self.apart]
        self.begin = [films[0] + ap for films,
                      ap in zip(self.flims_large, self.apart)]
        self.end = [films[1] - ap for films,
                    ap in zip(self.flims_large, self.apart)]

    def atten_func_full(self, fv):
        a_s, b_s = [], []
        for films, begin, end in zip(self.flims_large, self.begin, self.end):
            a_s.append(((self.amax - self.alpha) / (films[0] - begin),

                        (self.amax - self.alpha) / (films[1] - end)))
            b_s.append((-a_s[-1][0] * begin, -a_s[-1][1] * end))
        aten_large = np.zeros(fv.shape)
        for ii, f in enumerate(fv):
            aten = []

            for ff in f:

                if ff <= self.begin[ii]:
                    aten.append(a_s[ii][0] * ff + b_s[ii][0])
                elif ff >= self.end[ii]:
                    aten.append(a_s[ii][1] * ff + b_s[ii][1])
                else:
                    aten.append(0)
            aten = np.asanyarray(aten)
            aten_large[ii, :] = aten
        return aten_large + self.alpha

    def plot(self, fv):

        y = self.atten_func_full(fv)

        fig, ax = plt.subplots(1, 7, sharey=True, figsize=(20, 10))

        for f, yy, axn in zip(fv, y, ax):
            axn.plot(f, yy)
        plt.savefig(
            "loss_function_fibre.png", bbox_inches='tight')
        plt.close(fig)


class Noise(object):

    def __init__(self, int_fwm, sim_wind):
        self.pquant = np.array([np.sum(
            Planck*(fv*1e12)/(T*1e-12))
            for fv, T in zip(sim_wind.fv, sim_wind.T)])
        self.pquant = (self.pquant/2)**0.5
        return None

    def noise_func(self, int_fwm):
        seed = np.random.seed(int(time()*np.random.rand()))
        noise = np.array([pquant * (np.random.randn(int_fwm.nt) +
                                    1j*np.random.randn(int_fwm.nt))
                        for pquant in self.pquant])

        return noise

    def noise_func_freq(self, int_fwm, sim_wind):
        noise = self.noise_func(int_fwm)
        noise_freq = fftshift(fft(noise), axes=-1)
        return noise_freq


class Raman_banded(object):
    def __init__(self, sim_wind, sim_wind_g, how='load'):
        self.how = how
        if self.how == 'load':
            self.get_raman = self.raman_load
        else:
            self.get_raman = self.raman_analytic
        self.t = sim_wind_g.t + np.abs(sim_wind_g.t.min())
        self.fv_ram = sim_wind_g.fv - sim_wind_g.fp
        self.dt = sim_wind.dt[0]
        self.F = sim_wind.F

    def set_raman_band(self):
        hfmeas = self.get_raman()
        hfmeas_func_r = InterpolatedUnivariateSpline(self.fv_ram, hfmeas.real)
        hfmeas_func_i = InterpolatedUnivariateSpline(self.fv_ram, hfmeas.imag)

        raman_band_factors_re = [hfmeas_func_r(i*self.F)
                                 for i in (-2, -1, 1, 2)]

        raman_band_factors_im = [hfmeas_func_i(i*self.F)
                                 for i in (-2, -1, 1, 2)]
        raman_band_factors = [(i + 1j * j)*self.dt for i, j in
                              zip(raman_band_factors_re,
                                  raman_band_factors_im)]
        return raman_band_factors

    def raman_load(self):

        mat = loadmat('loading_data/silicaRaman.mat')
        ht = mat['ht']
        t1 = mat['t1']
        htmeas_func = InterpolatedUnivariateSpline(t1*1e-3, ht)
        htmeas = htmeas_func(self.t)
        htmeas *= (self.t > 0)*(self.t < 1)
        htmeas /= (self.dt * np.sum(htmeas))
        hfmeas = fftshift(fft(htmeas))
        return hfmeas

    def raman_analytic(self, sim_wind):
        t11 = 12.2e-3     # [ps]
        t2 = 32e-3        # [ps]
        # analytical response
        ht = (t11**2 + t2**2)/(t11*t2**2) * \
            np.exp(-t/t2*(self.t >= 0))*np.sin(self.t/t11)*(self.t >= 0)
        ht_norm = ht / (self.dt * np.sum(ht))
        htmeas = fftshift(fft(ht_norm))
        return htmeas
