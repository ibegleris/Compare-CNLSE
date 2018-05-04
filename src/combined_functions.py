import numpy as np
from scipy.fftpack import fft, ifft
from scipy.constants import c, pi
import sys
from numpy.fft import fftshift
from math import factorial
from integrands import *
from RK_methods import *
from time import time
import csv
import os


def pulse_propagation_constant(u, U, int_fwm, sim_wind, Dop, dAdzmm, z_vec):
    chunk = 0.1
    dz = z_vec.step

    u_large = np.zeros(
        [int(z_vec.end/chunk)+1, 3, u.shape[-1]], dtype=np.complex128)
    z_photo_vec = np.empty([int(z_vec.end/chunk)+2])
    u_large[0, :, :] = np.copy(u)
    z_photo_vec = np.array([i*chunk for i in range(u_large.shape[0])])
    j = 1
    print('endis', z_vec.end)
    disp = np.exp(Dop*dz/2)
    ranging = z_vec.xfrange()
    next(ranging)
    for z in ranging:
        u = ifft(disp * fft(u))
        u = RK4(dAdzmm, u, dz)
        u = ifft(disp * fft(u))
        if np.allclose(z_photo_vec[j], z, rtol=0, atol=1e-08):
            print(j, z)
            u_large[j, :, :] = np.copy(u)
            j += 1

    if not(np.allclose(z, z_vec.end)):
        dz_extra = z_vec.end - z
        disp = np.exp(0*dz_extra/2)
        u = ifft(disp * fft(u))
        u = RK4(dAdzmm, u, dz_extra)
        u = ifft(disp * fft(u))
        u_large[j, :, :] = u
    U_large = fftshift(fft(u_large), axes=-1)
    U = fftshift(fft(u), axes=-1)
    return u, U, u_large, U_large, z_photo_vec


def pulse_propagation_constant_timer(u, U, int_fwm, sim_wind,
                                     Dop, dAdzmm, z_vec):
    dz = z_vec.step
    t_vec = np.empty(100)

    for kk in range(len(t_vec)):
        ranging = z_vec.xfrange()
        next(ranging)
        t = time()
        disp = np.exp(Dop*dz/2)
        for z in ranging:
            u = ifft(disp * fft(u))
            u = RK4(dAdzmm, u, dz)
            u = ifft(disp * fft(u))
        if not(np.allclose(z, z_vec.end)):
            dz_extra = z_vec.end - z
            disp = np.exp(Dop*dz_extra/2)
            u = ifft(disp * fft(u))
            u = RK4(dAdzmm, u, dz_extra)
            u = ifft(disp * fft(u))
        t_vec[kk] = time() - t
    t_av = np.average(t_vec)
    t_std = np.std(t_vec)
    if sim_wind.type == 'BNLSE':
        df = sim_wind.df[0]
    else:
        df = sim_wind.df
    with open('timing/'+sim_wind.type+'.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([df, t_av, t_std])

    U = fftshift(fft(u), axes=-1)

    return u, U, 0, 0, 0


def pulse_propagation_adaptive(u, U, int_fwm, sim_wind,
                               Dop, dAdzmm, z_vec=None):
    """Pulse propagation through optical fibre. Uses adaptive step method
       to achieve this. """
    dztot = 0  # total distance traveled
    Safety = 0.95  # Safety set by the adaptive stepper.
    u1 = np.ascontiguousarray(u)
    dz = int_fwm.dz * 1

    converged = False
    while not(converged):
        delta = 2*int_fwm.maxerr
        while delta > int_fwm.maxerr:
            u1new = ifft(np.exp(Dop*dz/2) * fft(u1))
            A, delta = RK45CK(dAdzmm, u1new, dz)
            if (delta > int_fwm.maxerr):
                # calculate the step (shorter) to redo
                dz *= Safety*(int_fwm.maxerr/delta)**0.25
        # ###############Successful step############## #
        # propagate the remaining half step
        u1 = ifft(np.exp(Dop*dz/2) * fft(A))
        dztot += dz
        try:
            dz = np.min(
                [Safety*dz*(int_fwm.maxerr/delta)**0.2, Safety*int_fwm.dzstep])
        except RuntimeWarning:
            dz = Safety*int_fwm.dzstep
        ###################################################################

        if dztot == (int_fwm.dzstep):
            converged = True
        elif ((dztot + dz) >= int_fwm.dzstep):
            dz = int_fwm.dzstep - dztot
        ###################################################################

    u = u1
    U = fftshift(fft(u), axes=-1)
    int_fwm.dz = dz*1
    return u, U, 0, 0, 0


def dispersion_operator(betas, lamda_c, int_fwm, sim_wind):
    """
    Calculates the dispersion operator in rad/m units
    INputed are the dispersion operators at the omega0
    Local include the taylor expansion to get these opeators at omegac
    Returns Dispersion operator
    """
    c_norm = c*1e-12  # Speed of light [m/ps] #Central wavelength [nm]
    wc = 2*pi * c_norm / sim_wind.lamda
    w0 = 2*pi * c_norm / lamda_c

    betap = np.zeros_like(betas)

    for j in range(len(betas.T)):
        if j == 0:
            betap[j] = betas[j]
        fac = 0
        for k in range(j, len(betas.T)):
            betap[j] += (1/factorial(fac)) * \
                betas[k] * (wc - w0)**(fac)
            fac += 1
    w = sim_wind.w + sim_wind.woffset
    if len(w.shape) > 1:
        beta1 = []
        beta2 = []
        for i in range(-w.shape[0]//2+1, w.shape[0]//2 + 1):
            beta1.append(np.sum([betap[1] + betap[n] * (i *
                        sim_wind.Omega)**(n - 1) / factorial(n - 1)
                                 for n in range(2, len(betap))]))
            beta2.append(np.sum([betap[2]] + [betap[n] * (i *
                        sim_wind.Omega)**(n - 1) / factorial(n - 1)
                        for n in range(3, len(betap))]))
    else:
        betap[1] = 0

    Dop = np.zeros(sim_wind.fv.shape, dtype=np.complex128)
    alpha = np.reshape(int_fwm.alpha*0, np.shape(Dop))
    Dop -= fftshift(alpha/2)
    betap[0] -= betap[0]
    betap[1] -= betap[1]

    try:
        np.savetxt('output/data/betap.txt', betap)
    except FileNotFoundError:
        os.system('mkdir output')
        os.system('mkdir output/data/')
        np.savetxt('output/data/betap.txt', betap)
    for j, bb in enumerate(betap[2:]):
        Dop -= 1j*(w**(j+2) * bb / factorial(j+2))
    return Dop


class sim_parameters(object):
    def __init__(self, n2, nm, alphadB, betas, M, fr, T0):
        self.n2 = n2
        self.nm = nm
        self.alphadB = alphadB
        self.betas = betas
        self.M = M
        self.fr = fr
        self.T0 = T0

    def general_options(self, maxerr,
                        ss='1'):
        self.maxerr = maxerr
        self.ss = ss
        return None

    def propagation_parameters(self, N, z, dz_less):
        self.N = N
        self.nt = 2**self.N
        self.z = z
        self.dzstep = self.z
        self.dz = self.dzstep/dz_less
        return None


def overlap(n2, lamda_g, gama):
    """ Calculates the 1/Aeff (M) from the gamma given.
        The gamma is supposed to be measured at lamda_g
        (in many cases we assume that is the same as where
        the dispersion is measured at).
    """
    M = gama / (n2*(2*pi/lamda_g))
    return M


def dF_sidebands(beta, lamp, lam_z, n2, M, P, F_over=0, DF_over=0):

    omegap, omega_z = (1e-12*2*pi*c/i for i in (lamp, lam_z))
    if F_over != 0:
        print("WARNING: Using overwrite of Frequency bands!!!!!")
        return F_over, omegap/(2*pi)
    omega = omegap - omega_z
    gama = 1e12*n2*omegap/(c * (1/M))

    a = beta[4]/12 + omega * beta[5]/12
    b = beta[2] + omega * beta[3] + omega**2 * \
        beta[4] / 2 + omega**3 * beta[5]/6
    g = 2 * gama * P
    det = b**2 - 4 * a * g

    if det < 0:
        print('No sidebands predicted by simple model!')
        sys.exit(1)
    Omega = np.array([(-b + det**0.5) / (2*a), (-b - det**0.5) / (2*a)])
    Omega = Omega[Omega > 0]
    Omega = [i**0.5 for i in Omega]
    Omega = (Omega * np.logical_not(np.iscomplex(Omega))).real
    if len(Omega) == 0:
        print('Warning! No sideband.')
        print('Overwrite the procedure with your own F.')
        sys.exit(1)
    F = np.max(Omega) / (2*pi) + DF_over
    f_p = omegap/(2*pi)
    return F, f_p


def check_ft_grid(Fv, diff):
    """Grid check for fft optimisation"""
    def check(fv, diff):
        if fv.any() < 0:
            sys.exit("some of your grid is negative")

        lvio = []
        for i in range(len(fv)-1):
            lvio.append(fv[i+1] - fv[i])

        grid_error = np.abs(np.asanyarray(lvio)[:]) - np.abs(diff)
        if not(np.allclose(grid_error, 0, rtol=0, atol=1e-12)):
            print(np.max(grid_error))
            sys.exit("your grid is not uniform")

        if not(np.log2(np.shape(fv)[0]) == int(np.log2(np.shape(fv)[0]))):
            print("fix the grid for optimization\
                 of the fft's, grid:" + str(np.shape(fv)[0]))
            sys.exit(1)
    if len(Fv.shape) > 1:
        for fv in Fv:
            check(fv, diff)
    else:
        check(Fv, diff)
    return 0


"----------------------------------------------------------------------------"


"------------------------suplementary functions------------------------"


class float_range(object):
    def __init__(self, start, end, step):
        self.start = start
        self.end = end
        self.step = step

    def xfrange(self, decimals=6):
        i = 0
        while round(self.start + i * self.step, decimals) <= self.end:
            yield self.start + i * self.step
            i += 1


def my_arange(a, b, dr, decimals=6):
    res = np.array([a])
    k = 1
    while res[-1] < b:
        tmp = round(a + k*dr, decimals)
        if tmp > b:
            break
        res = np.append(res, tmp)
        k += 1
    return np.asarray(res)


def dbm2w(dBm):
    """This function converts a power given in dBm to a power given in W.
       Inputs::
               dBm(float): power in units of dBm
       Returns::
               Power in units of W (float)
    """
    return 1e-3*10**((dBm)/10.)


def w2dbm(W, floor=-100):
    """This function converts a power given in W to a power given in dBm.
       Inputs::
               W(float): power in units of W
       Returns::
               Power in units of dBm(float)
    """
    if type(W) != np.ndarray:
        if W > 0:
            return 10. * np.log10(W) + 30
        elif W == 0:
            return floor
        else:
            print(W)
            raise(ZeroDivisionError)
    a = 10. * (np.ma.log10(W)).filled(floor/10-3) + 30
    return a


"-------------------------------------------------------------------"
