import numpy as np
from numba import jit
def RK45CK(dAdzmm, u1, dz):
    """
    Propagates the nonlinear operator for 1 step using a 5th order Runge
    Kutta method and the cash karp algorithm.
    use: [A delta] = RK5mm(u1, dz)
    where u1 is the initial time vector
    hf is the Fourier transform of the Raman nonlinear response time
    dz is the step over which to propagate

    in output: A is new time vector
    delta is the norm of the maximum estimated error between a 5th
    order and a 4th order integration
    """
    A1 = dz*dAdzmm(u1)
    A2 = dz*dAdzmm(u1 + (1./5)*A1)
    A3 = dz*dAdzmm(u1 + (3./40)*A1 + (9./40)*A2)
    A4 = dz*dAdzmm(u1 + (3./10)*A1 - (9./10)*A2 + (6./5)*A3)
    A5 = dz*dAdzmm(u1 - (11./54)*A1 + (5./2)*A2 - (70./27)*A3 + (35./27)*A4)
    A6 = dz*dAdzmm(u1 + (1631./55296)*A1 + (175./512)*A2 + (575./13824)*A3 +
                        (44275./110592)*A4 + (253./4096)*A5)

    A = u1 + (37./378)*A1 + (250./621)*A3 + (125./594) * \
        A4 + (512./1771)*A6  # Fifth order accuracy

    Afourth = u1 + (2825./27648)*A1 + (18575./48384)*A3 + (13525./55296) * \
        A4 + (277./14336)*A5 + (1./4)*A6  # Fourth order accuracy
    delta = np.max(np.linalg.norm(A - Afourth, 2, axis=-1))
    return A, delta


def RK45DP(dAdzmm, u1, dz):
    A1 = dz*dAdzmm(u1)
    A2 = dz*dAdzmm(u1 + (1./5)*A1)
    A3 = dz*dAdzmm(u1 + (3./40)*A1 + (9./40)*A2)
    A4 = dz*dAdzmm(u1 + (44./45)*A1 - (56./15)*A2 + (32./9)*A3)
    A5 = dz*dAdzmm(u1 + (19372./6561)*A1 - (25360./2187)*A2 +
                       (64448./6561)*A3 - (212./729)*A4)
    A6 = dz*dAdzmm(u1 + (9017./3168)*A1 - (355./33)*A2 + (46732./5247)*A3 +
                       (49./176)*A4 - (5103./18656)*A5)
    A = u1 + (35./384)*A1 + (500./1113)*A3 + (125./192) * \
        A4 - (2187./6784)*A5 + (11./84)*A6
    A7 = dz*dAdzmm(A)

    Afourth = u1 + (5179/57600)*A1 + (7571/16695)*A3 + (393/640)*A4 - \
        (92097/339200)*A5 + (187/2100)*A6 + (1/40)*A7  # Fourth order accuracy

    delta = np.max(np.linalg.norm(A - Afourth, 2, axis=-1))
    return A, delta


@jit(nogil=True)
def RK4(dAdzmm, u1, dz):
    k1 = dz * dAdzmm(u1)
    k2 = dz * dAdzmm(u1 + 0.5 * k1)
    k3 = dz * dAdzmm(u1 + 0.5 * k2)
    k4 = dz * dAdzmm(u1 + k3)
    return  u1 + (1/6) * ( k1 + 2 *k2 + 2 * k3 + k4) 