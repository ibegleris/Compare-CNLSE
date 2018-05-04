import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.fftpack import fft
from combined_functions import check_ft_grid
from scipy.constants import pi, c, hbar
from numpy.fft import fftshift
from scipy.io import loadmat
from time import time
import sys
import matplotlib.pyplot as plt
from scipy.integrate import simps


def fv_creator(fp, df, F, int_fwm):
    """
    Cretes frequency grid such that the estimated MI-FWM bands
    will be on the grid and extends this such that to avoid
    fft boundary problems.
    Inputs::
        lamp: wavelength of the pump (float)
        lamda_c: wavelength of the zero dispersion wavelength(ZDW) (float)
        int_fwm: class that holds nt (number of points in each band)
        betas: Taylor coeffiencts of beta around the ZDW (Array)
        M : The M coefficient (or 1/A_eff) (float)
        P_p: pump power
        Df_band: band frequency bandwidth in Thz, (float)
    Output::
        fv: Frequency vector of bands (Array of shape [nt])
    """
    f_centrals = [fp + i * F for i in range(-1, 2)]
    fv1 = np.linspace(f_centrals[0], f_centrals[1],
                      int_fwm.nt//4 - 1, endpoint=False)
    df = fv1[1] - fv1[0]
    fv2 = np.linspace(f_centrals[1], f_centrals[2], int_fwm.nt//4)
    try:
        assert df == fv2[1] - fv2[0]
    except AssertionError:
        print(df, fv2[1] - fv2[0])
    fv0, fv3 = np.zeros(int_fwm.nt//4 + 1), np.zeros(int_fwm.nt//4)
    fv0[-1] = fv1[0] - df
    fv3[0] = fv2[-1] + df

    for i in range(1, len(fv3)):
        fv3[i] = fv3[i - 1] + df
    for i in range(len(fv0) - 2, -1, -1):
        fv0[i] = fv0[i + 1] - df

    assert not(np.any(fv0 == fv1))
    assert not(np.any(fv1 == fv2))
    assert not(np.any(fv2 == fv3))
    fv = np.concatenate((fv0, fv1, fv2, fv3))
    for i in range(3):
        assert f_centrals[i] in fv

    check_ft_grid(fv, df)
    p_pos = np.where(np.abs(fv - fp) == np.min(np.abs(fv - fp)))[0]
    return fv, p_pos, f_centrals


class raman_object(object):
    """
    Warning: hf comes back normalised but ht does not!!!
    """

    def __init__(self, b=None):
        self.how = b
        self.hf = None
        self.ht = None

    def raman_load(self, t, dt):

        if self.how == 'analytic':
            t11 = 12.2e-3     # [ps]
            t2 = 32e-3       # [ps]
            # analytical response
            ht = (t11**2 + t2**2)/(t11*t2**2) * \
                np.exp(-t/t2*(t >= 0))*np.sin(t/t11)*(t >= 0)
            self.ht = ht  # * dt
            ht_norm = ht / simps(ht, t)

            # Fourier transform of the analytic nonlinear response
            self.hf = fft(ht_norm)
        elif self.how == 'load':
            # loads the measured response (Stolen et al. JOSAB 1989)
            mat = loadmat('loading_data/silicaRaman.mat')
            ht = mat['ht']
            t1 = mat['t1']
            htmeas_func = InterpolatedUnivariateSpline(t1*1e-3, ht)
            ht = htmeas_func(t)
            ht *= (t > 0)*(t < 1)  # only measured between +/- 1 ps)
            self.ht = ht / simps(ht, t)
            ht_norm = ht / simps(ht, t)
            # Fourier transform of the measured nonlinear response

            self.hf = fft(ht_norm)
        else:
            sys.exit("No raman response on the GNLSE")
        return None


class sim_window(object):

    def __init__(self, fv, lamda, F, lamda_c, int_fwm, where):
        self.fv = fv
        self.type = 'GNLSE'
        self.lamda = lamda
        self.fp = 1e-12*c/self.lamda
        self.fmed = 0.5*(fv[-1] + fv[0])*1e12  # [Hz]
        self.deltaf = np.max(self.fv) - np.min(self.fv)  # [THz]
        self.df = self.deltaf/int_fwm.nt  # [THz]
        self.T = 1 / self.df  # Time window (period)[ps]
        self.woffset = 2*pi*(self.fmed - c/lamda)*1e-12  # [rad/ps]

        self.w0 = 2*pi*self.fmed  # central angular frequency [rad/s]

        self.tsh = (1/self.w0)*1e12  # shock time [ps]
        self.dt = self.T/int_fwm.nt  # timestep (dt)     [ps]
        # time vector      [ps]
        self.t = (range(int_fwm.nt)-np.ones(int_fwm.nt)*int_fwm.nt/2)*self.dt
        self.w = fftshift(2*pi * (self.fv - 1e-12*self.fmed))
        self.t_band = self.t

        self.lv = 1e-3*c/self.fv
        self.zv = int_fwm.dzstep*np.asarray(range(0, 2))
        self.p_pos = where
        self.F = F
        self.f_centrals = np.array(
            [1e-12 * c/lamda + i * F for i in range(-1, 2)])
        self.w_tiled = fftshift(
            2*pi * (self.fv - self.f_centrals[1]))  # w of self-step


class Loss(object):

    def __init__(self, int_fwm, sim_wind, amax=None, apart_div=8):
        """
        Initialise the calss Loss, takes in the general parameters and
        the freequenbcy window. From that it determines where
        the loss will become freequency dependent. With the default value
        being an 8th of the difference of max and min.

        """
        self.alpha = int_fwm.alphadB/4.343
        if amax is None:
            self.amax = self.alpha
        else:
            self.amax = amax/4.343

        self.flims_large = (np.min(sim_wind.fv), np.max(sim_wind.fv))
        try:
            self.begin = apart_div[0]
            self.end = apart_div[1]
        except TypeError:

            self.apart = np.abs(self.flims_large[1] - self.flims_large[0])
            self.apart /= apart_div
            self.begin = self.flims_large[0] + self.apart
            self.end = self.flims_large[1] - self.apart

    def atten_func_full(self, fv):
        aten = []

        a_s = ((self.amax - self.alpha) / (self.flims_large[0] - self.begin),

               (self.amax - self.alpha) / (self.flims_large[1] - self.end))
        b_s = (-a_s[0] * self.begin, -a_s[1] * self.end)

        for f in fv:
            if f <= self.begin:
                aten.append(a_s[0] * f + b_s[0])
            elif f >= self.end:
                aten.append(a_s[1] * f + b_s[1])
            else:
                aten.append(0)
        return np.asanyarray(aten) + self.alpha

    def plot(self, fv):
        fig = plt.figure()
        y = self.atten_func_full(fv)
        plt.plot(fv, y)
        plt.xlabel("Frequency (Thz)")
        plt.ylabel("Attenuation (cm -1 )")
        plt.savefig(
            "loss_function_fibre.png", bbox_inches='tight')
        plt.close(fig)


class Noise(object):

    def __init__(self, int_fwm, sim_wind):
        self.pquant = np.sum(
            hbar*(sim_wind.w*1e12 + sim_wind.w0)/(sim_wind.T*1e-12))
        self.pquant = (self.pquant/2)**0.5
        return None

    def noise_func(self, int_fwm):
        seed = np.random.seed(int(time()*np.random.rand()))
        noise = self.pquant * (np.random.randn(int_fwm.nt) +
                                1j*np.random.randn(int_fwm.nt))
        return noise

    def noise_func_freq(self, int_fwm, sim_wind):
        noise = self.noise_func(int_fwm)
        noise_freq = fftshift(fft(noise))
        return noise_freq
