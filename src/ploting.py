import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from combined_functions import w2dbm
import h5py


class Plot2d(object):
    def __init__(self, Ui, sim_wind, eqtype, extra_name='', Deltaf=0.4):
        if len(Ui.shape) > 1:
            self.dt = sim_wind.dt[0]
            self.Ui = \
                (self._reshape(i) for i in (Ui,))
            self.Ui = Ui
            self.t = sim_wind.t
            self.plot_f = self.plot_3_spec
            self.fv = sim_wind.fv
        else:
            self.dt = sim_wind.dt
            self.Ui, self.fv, self.t = \
                Ui, sim_wind.fv, sim_wind.t
            self.plot_f = self.plot_1_spec
        self.Ui_max = np.max(w2dbm(self.dt**2 * np.abs(self.Ui)**2))

        self.eqtype = eqtype
        self.fig_folder = 'output/figures/' + extra_name
        self.dat_folder = 'output/data/' + extra_name
        self.D_init = {'t': self.t, 'f': self.fv}
        self.sim_wind = sim_wind
        self.Deltaf = Deltaf

    def _reshape(self, x):
        return np.reshape(x, x.shape[0]*x.shape[1])

    def reshape(self, u):
        if len(u.shape) > 1:
            u = self._reshape(u)
        return u

    def plot_t(self, u, pos):
        u = self.reshape(u)
        t = self.reshape(self.t)
        fig = plt.figure(figsize=(20, 5))
        plt.plot(t, np.abs(u)**2)
        plt.xlabel('t (ps)')
        plt.ylabel('Spec (W)')
        plt.savefig(self.fig_folder+self.eqtype+'_time_'+pos,
                    bbox_inches='tight')
        # plt.show()
        plt.close()
        return None

    def plot_3_spec(self, U, pos):
        self.plot_1_spec_large(U, pos)
        U = w2dbm(self.dt**2 * np.abs(U)**2)
        U -= self.Ui_max
        f, AX = plt.subplots(1, 3, sharey=True, figsize=(20, 5))
        for i, ax in enumerate(AX):
            ax.plot(self.fv[i, :], U[i, :])
            ax.set_xlim([self.sim_wind.f_centrals[i] - self.Deltaf,
                         self.sim_wind.f_centrals[i] + self.Deltaf])
            ax.set_ylim([-120, 0])
            ax.xaxis.grid(True, zorder=0.5)
            ax.yaxis.grid(True, zorder=0.5)
        AX[1].set_title(self.eqtype)
        plt.savefig(self.fig_folder+self.eqtype+'_freq_'+pos,
                    bbox_inches='tight')
        # plt.show()
        plt.close()
        return None

    def plot_1_spec(self, U, pos):
        self.plot_1_spec_large(U, pos)
        U = w2dbm(self.dt**2 * np.abs(U)**2)
        U -= self.Ui_max

        f, AX = plt.subplots(1, 3, sharey=True, figsize=(20, 5))
        for i, ax in enumerate(AX):
            ax.plot(self.fv, U)
            ax.set_xlim([self.sim_wind.f_centrals[i] - self.Deltaf,
                         self.sim_wind.f_centrals[i] + self.Deltaf])
            ax.set_ylim([-120, 0])
            ax.xaxis.grid(True, zorder=0.5)
            ax.yaxis.grid(True, zorder=0.5)
        AX[1].set_title(self.eqtype)
        plt.savefig(self.fig_folder+self.eqtype+'_freq_'+pos,

                    bbox_inches='tight')
        # plt.show()
        plt.close()
        return None

    def plot_1_spec_large(self, U, pos):
        U = w2dbm(self.dt**2 * np.abs(self.reshape(U))**2)
        if len(self.fv.shape) > 1:
            fv = self.reshape(self.fv)
        else:
            fv = self.fv
        U -= self.Ui_max
        fig = plt.figure(figsize=(10, 5))
        plt.plot(fv, U, linewidth=2)
        plt.xlabel('f (Thz)')
        plt.ylabel('Spec (AU)')

        #plt.xlim([284.25, 286.25])
        plt.savefig(self.fig_folder+self.eqtype+'_freq_large_'+pos,
                    bbox_inches='tight')
        # plt.show()
        plt.close()
        return None

    def save_data(self, pos, u, U, u_large=0, U_large=0,
                  z_photo_vec=0, gama=0, woffset=0):
        u, U = self.reshape(u), self.reshape(U)
        U -= self.Ui_max
        D = {**self.D_init, **{'u': u, 'U': U, 'u_large': u_large,
                               'U_large': U_large, 'z_photo_vec': z_photo_vec,
                               'gama': gama, 'woffset': woffset}}

        filename = self.dat_folder+'data_'+self.eqtype+'_'+pos
        save_hdf5(filename, **D)
        return None


def save_hdf5(filename, **variables):
    with h5py.File(filename + '.hdf5', 'a') as f:
        for i in (variables):
            f.create_dataset(str(i), data=variables[i])
    return None


def read_variables(filename):
    with h5py.File(filename+'.hdf5', 'r') as f:
        D = {}
        for i in f.keys():
            try:
                D[str(i)] = f.get(str(i)).value
            except AttributeError:
                pass
    return D
