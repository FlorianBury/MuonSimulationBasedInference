import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from tools import *
from preprocessing import *
from plotting import *

class MuonDataset(Dataset):
    def __init__(
        self,
        ri,
        rf,
        di,
        df,
        pi,
        selection = None,
    ):
        self.ri = ri
        self.rf = rf
        self.di = di
        self.df = df
        self.pi = pi

        self.selection = selection

        # Find the wall coordinates
        self.low = torch.minimum(
            self.ri.min(dim=0).values,
            self.rf.min(dim=0).values,
        )
        self.high = torch.maximum(
            self.ri.max(dim=0).values,
            self.rf.max(dim=0).values,
        )


        # Mask #
        self.selection = selection
        if self.selection is not None:
            self.mask = selection(
                ri = self.ri,
                rf = self.rf,
                di = self.di,
                df = self.df,
            )
            #print (f'Masking {self.mask.numel()-sum(self.mask)} out of {self.mask.numel()} events [{(self.mask.numel()-sum(self.mask))/self.mask.numel()*100:5.3f}%]')
        else:
            self.mask = torch.fill(self.ri.shape[0],fill_value=True)


        # Get observables and conditions #
        a1,a2,s1,s2 = positions_to_observables(
            ri = self.ri,
            di = self.di,
            rf = self.rf,
            df = self.df,
            lo = self.low,
            hi = self.high,
        )
        self._observables = torch.stack([a1, a2, s1, s2], dim=-1)
        self._conditions = torch.cat(
            [
                self.ri,
                self.di,
                torch.log(self.pi),
            ],
            dim = -1,
        )
        self.observables = self._observables[self.mask]
        self.conditions = self._conditions[self.mask]
        self.observable_names = [r'$\Delta\theta_1$',r'$\Delta\theta_2$',r'$\Delta r_1$',r'$\Delta r_2$',]
        self.condition_names = ['x','y','z',r'$d_x$',r'$d_y$',r'$d_z$',r'$\log P$']

    @classmethod
    def from_data_file(
        cls,
        data_file,
        momentum_file,
        selection = None,
    ):
        # Load data
        data = np.load(data_file)
        pi = torch.tensor(np.load(momentum_file), dtype=torch.float32).reshape(-1,1) / 1000 # GeV

        # Split into positions and directions
        ri = torch.tensor(data[:, 0:3], dtype=torch.float32)
        di = torch.tensor(data[:, 3:6], dtype=torch.float32)
        rf = torch.tensor(data[:, 6:9], dtype=torch.float32)
        df = torch.tensor(data[:, 9:12], dtype=torch.float32)
        #df = - df

        return cls(
            ri = ri,
            rf = rf,
            di = di,
            df = df,
            pi = pi,
            selection = selection,
        )

    def save(
        self,
        data_file,
        momentum_file,
        overwrite=False,
    ):
        if os.path.exists(data_file) and not overwrite:
            raise RuntimeError(f'{data_file} already exists')
        if os.path.exists(momentum_file) and not overwrite:
            raise RuntimeError(f'{momentum_file} already exists')
        data = np.concatenate(
            [
                self.ri,
                self.di,
                self.rf,
                self.df,
            ],
            axis = -1,
        )
        np.save(data_file,data)
        np.save(momentum_file,self.pi*1000) # MeV

    def get_observable_preprocessing(self):
        preprocessing = ObservablePreprocessing(c=1)
        preprocessing.fit(self.observables)
        return preprocessing

    def get_condition_preprocessing(self):
        preprocessing = ConditionPreprocessing(a=-1,b=1)
        preprocessing.fit(self.conditions)
        return preprocessing

    @property
    def observable_dim(self):
        return self.observables.shape[-1]

    @property
    def condition_dim(self):
        return self.conditions.shape[-1]

    def __len__(self):
        return self.observables.shape[0]

    def __getitem__(self, idx):
        return (
            self.observables[idx],
            self.conditions[idx],
        )

    def plot_3d(self, N=100, indices=None, arrow_scale=1., show_direction=False, show_path=False, seed=None):
        return plot_particles_3d(
            ri = self.ri,
            rf = self.rf,
            di = self.di,
            df = self.df,
            pi = self.pi,
            pf = self.pi, # TODO change when available
            low = self.low,
            high = self.high,
            N = N,
            indices = indices,
            arrow_scale = arrow_scale,
            show_direction = show_direction,
            show_path = show_path,
            seed = seed,
        )

    def plot_observables(self,log=False):
        fig = plt.figure()
        plt.title('Observables')
        for i,name in enumerate([r'$\Delta\theta_1$',r'$\Delta\theta_2$',r'$\Delta r_1$',r'$\Delta r_2$',]):
            plt.hist(
                self.observables[:,i],
                bins = 100,
                histtype = 'step',
                label = name,
            )
        if log:
            plt.yscale('log')
        plt.legend()
        return fig
