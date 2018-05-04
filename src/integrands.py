from scipy.fftpack import fft, ifft
from numpy.fft import fftshift
from scipy.constants import pi
import numpy as np
from numba import jit


class Integrand_GNLSE(object):
    def __init__(self, int_fwm, sim_wind):
        self.gama = -1j*int_fwm.n2*2*pi/sim_wind.lamda * int_fwm.M
        self.dt = sim_wind.dt
        self.w_tiled = sim_wind.w_tiled
        self.hf = sim_wind.hf
        self.tsh = sim_wind.tsh

        self.fr = int_fwm.fr
        self.kr = 1 - self.fr

        if int_fwm.ss == 1:
            self.dAdzmm = self.dAdzmm_s1
        else:
            self.dAdzmm = self.dAdzmm_s0

        if self.fr == 0:
            self.nonlinear_term = self.nonlinear_term_Kerr_only
        else:
            self.nonlinear_term = self.nonlinear_term_Raman

    def nonlinear_term_Raman(self, u0, M3):
        N = u0*(self.kr*M3 + self.fr * self.dt *
                fftshift(ifft(fft(M3) * self.hf)))
        return N

    @jit(nogil=True)
    def nonlinear_term_Kerr_only(self, u0, M3):
        N = u0*(self.kr*M3)
        return N

    @jit(nogil=True)
    def dAdzmm_s0(self, u0):
        M3 = u0.real * u0.real + u0.imag * u0.imag
        N = self.nonlinear_term(u0, M3)
        N = self.gama * N
        return N

    def dAdzmm_s1(self, u0):
        M3 = u0.real * u0.real + u0.imag * u0.imag
        N = self.nonlinear_term(u0, M3)
        N = self.gama * (N + self.tsh*ifft(self.w_tiled * fft(N)))
        return N


class Integrand_BNLSE(object):
    def __init__(self, int_fwm_b, sim_wind_b):
        self.gama = int_fwm_b.gama
        self.w_tiled = sim_wind_b.w_tiled
        self.tsh = sim_wind_b.tsh
        self.N = np.zeros([3, sim_wind_b.w_tiled.shape[-1]], dtype=np.complex)
        self.fr = int_fwm_b.fr
        self.kr = 1 - self.fr
        hf_2, hf_1, hf1, hf2 = [self.fr * i for i in sim_wind_b.hf]
        _xpm = 2 - self.fr
        self.factors_xpm = np.array([[1, _xpm + hf_1, _xpm + hf_2],
                                     [_xpm + hf1, 1, _xpm + hf_1],
                                     [_xpm + hf2, _xpm + hf1, 1]])
        self.factors_fwm = np.array([self.kr + hf_1,
                                     self.kr + 0.5 * (hf1 + hf_1),
                                     self.kr + hf1])
        if int_fwm_b.ss == 1:
            self.dAdzmm = self.dAdzmm_s1
        else:
            self.dAdzmm = self.dAdzmm_s0
        if self.fr == 0:
            self.XPM_SPM_FWM = self.XPM_SPM_FWM_fr0
        else:
            self.XPM_SPM_FWM = self.XPM_SPM_FWM_fr

    @jit(nogil=True)
    def XPM_SPM_FWM_fr0(self, u0, u0_c):
        u_pump2 = u0[1, :]**2
        u_abs2 = u0.real * u0.real + u0.imag * u0.imag
        N = np.zeros(u0.shape, dtype=np.complex128)

        # FWM
        N[0, :] = u_pump2 * u0_c[2, :]
        N[1, :] = 2 * u0[0, :] * u0[2, :] * u0_c[1, :]
        N[2, :] = u_pump2 * u0_c[0, :]

        # SPM-XPM
        N += np.matmul(self.factors_xpm, u_abs2) * u0
        return N

    @jit(nogil=True)
    def XPM_SPM_FWM_fr(self, u0, u0_c):
        u_pump2 = u0[1, :]**2
        u_abs2 = u0.real * u0.real + u0.imag * u0.imag
        N = np.zeros(u0.shape, dtype=np.complex128)

        # FWM
        N[0, :] = u_pump2 * u0_c[2, :]
        N[1, :] = 2 * u0[0, :] * u0[2, :] * u0_c[1, :]
        N[2, :] = u_pump2 * u0_c[0, :]
        N = self.factors_fwm[:, np.newaxis] * N

        # SPM-XPM
        N += np.matmul(self.factors_xpm, u_abs2) * u0
        return N

    def dAdzmm_s0(self, u0):
        u0_c = u0.conjugate()
        N = self.XPM_SPM_FWM(u0, u0_c)
        N = self.gama * N
        return N

    def dAdzmm_s1(self, u0):
        u0_c = u0.conjugate()
        N = self.XPM_SPM_FWM(u0, u0_c)
        stepn = ifft(self.w_tiled * fft(N))
        for i in range(3):
            N[i, :] = self.gama[i] * (N[i, :] + self.tsh[i] * stepn[i, :])
        return N
