import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader,Subset
import matplotlib
import matplotlib.pyplot as plt

from tools import *
from constants import *
from plotting import plot_3d_hist2d_faces


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 5,
        hidden_features: tuple = (128, 128),
        passes = None,
        bins: int = 8,
    ):
        """
        Args:
            features: Number of dimensions in the target variable x.
            context: Number of conditioning features. Use 0 for
                     an unconditional flow.
            transforms: Number of flow transformations.
            hidden_features: Hidden layer sizes for the flow networks.
            bins: Number of bins for the spline transformations.
        """
        super().__init__()

        self.features = features
        self.context_features = context

        self.flow = zuko.flows.NSF(
            features=features,
            context=context,
            transforms=transforms,
            hidden_features=hidden_features,
            passes=passes,
            bins=bins,
        )

    def distribution(self, context=None):
        """Return the learned probability distribution."""
        if self.context_features > 0:
            if context is None:
                raise ValueError(
                    "This is a conditional flow; context must be provided."
                )
            return self.flow(context)

        return self.flow()

    def forward(self, x, context=None):
        """
        Return log p(x) or log p(x | context).

        Args:
            x: Target tensor, shape (N, features).
            context: Optional conditioning tensor, shape (N, context_features).

        Returns:
            Log-probabilities, shape (N,).
        """
        return self.distribution(context).log_prob(x)

    def sample(self, n_samples, context=None):
        """
        Generate samples.

        For a conditional flow, context determines the distribution
        from which samples are generated.
        """
        dist = self.distribution(context)

        if self.context_features > 0:
            # One sample per requested draw, for every context row.
            return dist.sample((n_samples,))

        return dist.sample((n_samples,))


class LossSurrogate(nn.Module):
    def __init__(
        self,
        flow,
        device = None,
    ):
        super().__init__()

        self.flow = flow

        self.dataset = None
        self.obs_prep = None
        self.cond_prep = None
        self.train_indices = None
        self.valid_indices = None

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.01, weight_decay=0.)

        self.set_to_device(self.device)

    def init_dataset(self,dataset):
        self.dataset = dataset
        self.obs_prep = self.dataset.get_observable_preprocessing()
        self.cond_prep = self.dataset.get_condition_preprocessing()

        indices = np.arange(len(self.dataset))
        split = int(np.floor(0.9 * len(self.dataset)))
        np.random.shuffle(indices)
        self.train_indices, self.valid_indices = indices[:split], indices[split:]

    @property
    def train_dataset(self):
        return Subset(self.dataset,self.train_indices)

    @property
    def valid_dataset(self):
        return Subset(self.dataset,self.valid_indices)

    def set_to_device(self,device):
        self.device = device
        self.to(self.device)
        if hasattr(self, "optimizer"):
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)

    def train_model(
        self,
        epochs,
        lr,
        batch_size,
        num_workers = 0,
    ):
        if self.dataset is None:
            raise RuntimeError(f'No dataset, have you forgotten to call init_dataset ?')
        train_loader = DataLoader(
            dataset = self.train_dataset,
            batch_size = batch_size,
            shuffle = True,
            pin_memory = True,
            num_workers = num_workers,
        )
        valid_loader = DataLoader(
            dataset = self.valid_dataset,
            batch_size = batch_size,
            shuffle = False,
            pin_memory = True,
            num_workers = num_workers,
        )
        print (f'Epochs = {epochs} - LR = {lr:1.3e} - batch size = {batch_size} - Training set = {len(self.train_indices)} [{len(train_loader)} batches] - Validation set = {len(self.valid_indices)} [{len(valid_loader)} batches]')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.set_to_device(self.device)

        for epoch in range(epochs):
            # Training #
            self.flow.train()
            train_losses = torch.zeros(len(train_loader))
            for batch_idx, (obs,cond) in tqdm(enumerate(train_loader),desc='Training',leave=False,total=len(train_loader)):
                # Move to device #
                obs = obs.to(self.device)
                cond = cond.to(self.device)
                # Preprocess #
                obs = self.obs_prep(obs)
                cond = self.cond_prep(cond)
                # Loss #
                loss = - self.flow(obs,cond).mean()
                train_losses[batch_idx] = loss.item()
                # GD #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Validation #
            self.flow.eval()
            valid_losses = torch.zeros(len(valid_loader))
            for batch_idx, (obs,cond) in tqdm(enumerate(valid_loader),desc='Validation',leave=False,total=len(valid_loader)):
                # Move to device #
                obs = obs.to(self.device)
                cond = cond.to(self.device)
                # Preprocess #
                obs = self.obs_prep(obs)
                cond = self.cond_prep(cond)
                # Loss #
                with torch.no_grad():
                    loss = - self.flow(obs,cond).mean()
                valid_losses[batch_idx] = loss.item()

            print(f"Epoch: {epoch+1:3d}/{epochs} - Loss: {train_losses.mean():8.3f} - Val loss {valid_losses.mean():8.3f}")
        self.flow.eval()

    def set_gradients(self, flag: bool):
        for param in self.flow.parameters():
            param.requires_grad = flag

    def log_prob(self,obs,cond):
        # Move to device #
        obs = obs.to(self.device)
        cond = cond.to(self.device)
        # Preprocess #
        obs = self.obs_prep(obs)
        cond = self.cond_prep(cond)
        outside_mask = torch.logical_or(
            obs > 5.,
            obs < -5.,
        )
        if outside_mask.sum()>0:
            print (f'Values outside [-5,+5] : {outside_mask.sum()/len(outside_mask)*100:5.3f}%')
        # Log prob #
        return self.flow(obs,cond)

    @torch.no_grad()
    def log_prob_dataset(self,dataset,batch_size):
        loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = False,
        )
        log_probs = []
        for obs,cond in tqdm(loader,total=len(loader),desc='Log prob',leave=True):
            log_probs.append(self.log_prob(obs,cond).cpu())
        return torch.cat(log_probs)

    def validate_model(
        self,
        batch_size,
        n_samples = 1,
    ):
        valid_loader = DataLoader(
            dataset = Subset(self.dataset,self.valid_indices),
            batch_size = batch_size,
            shuffle = False,
            pin_memory = True,
        )

        true_obs = []
        samp_obs = []
        for batch_idx, (obs,cond) in tqdm(enumerate(valid_loader),desc='Validation',leave=False,total=len(valid_loader)):
            # Move to device #
            cond = cond.to(self.device)
            # Preprocess #
            cond = self.cond_prep(cond)
            with torch.no_grad():
                if cond.shape[-1] == 0:
                    samp = self.flow.sample(
                        n_samples = cond.shape[0],
                        context = cond,
                    )
                else:
                    samp = self.flow.sample(
                        n_samples = n_samples,
                        context = cond,
                    )[0].cpu()
            # Record #
            #obs = obs.unsqueeze(0).repeat_interleave(n_samples,dim=0).reshape(-1,obs.shape[-1])
            true_obs.append(obs)
            samp_obs.append(self.obs_prep.inverse(samp).cpu())


        true_obs = torch.cat(true_obs,dim=0)
        samp_obs = torch.cat(samp_obs,dim=0)

        fig,axs = plt.subplots(
            ncols = true_obs.shape[1],
            nrows = 1,
            figsize=(6*true_obs.shape[1],4),
        )
        plt.subplots_adjust(wspace=0.4)
        for i in range(true_obs.shape[1]):
            bins = np.linspace(
                min(true_obs[:,i].min(),samp_obs[:,i].min()),
                max(true_obs[:,i].max(),samp_obs[:,i].max()),
                50,
            )
            axs[i].hist(
                true_obs[:,i].numpy(),
                bins = bins,
                histtype = 'step',
                label = 'True observable',
                color = 'royalblue',
            )
            axs[i].hist(
                samp_obs[:,i].numpy(),
                bins = bins,
                histtype = 'step',
                label = 'Sampled observable',
                color = 'orange',
            )
            axs[i].legend(fontsize=10)
            axs[i].set_xlabel(self.dataset.observable_names[i],fontsize=14)
            axs[i].set_yscale('log')
        plt.show()

#        fig_true,axs_true = plt.subplots(
#            ncols = true_obs.shape[1]-1,
#            nrows = true_obs.shape[1]-1,
#            figsize=(5*true_obs.shape[1],4*true_obs.shape[1]),
#        )
#        plt.subplots_adjust(wspace=0.4,hspace=0.4)
#        fig_samp,axs_samp = plt.subplots(
#            ncols = true_obs.shape[1]-1,
#            nrows = true_obs.shape[1]-1,
#            figsize=(5*true_obs.shape[1],4*true_obs.shape[1]),
#        )
#        plt.subplots_adjust(wspace=0.4,hspace=0.4)
#        for i in range(true_obs.shape[1]-1):
#            for j in range(true_obs.shape[1]-1):
#                if i < j:
#                    axs_true[i,j].axis('off')
#                    axs_samp[i,j].axis('off')
#                else:
#                    bins_x = np.linspace(
#                        min(true_obs[:,i+1].min(),samp_obs[:,i+1].min()),
#                        max(true_obs[:,i+1].max(),samp_obs[:,i+1].max()),
#                        50,
#                    )
#                    bins_y = np.linspace(
#                        min(true_obs[:,j].min(),samp_obs[:,j].min()),
#                        max(true_obs[:,j].max(),samp_obs[:,j].max()),
#                        50,
#                    )
#                    H = axs_true[i,j].hist2d(
#                        true_obs[:,i+1].numpy(),
#                        true_obs[:,j].numpy(),
#                        bins = (bins_x,bins_y),
#                        norm = matplotlib.colors.LogNorm(),
#                    )
#                    axs_true[i,j].set_title('True observable')
#                    axs_true[i,j].set_xlabel(self.dataset.observable_names[i+1])
#                    axs_true[i,j].set_ylabel(self.dataset.observable_names[j])
#                    fig_true.colorbar(H[3],ax=axs_true[i,j])
#
#                    H = axs_samp[i,j].hist2d(
#                        samp_obs[:,i+1].numpy(),
#                        samp_obs[:,j].numpy(),
#                        bins = (bins_x,bins_y),
#                        norm = matplotlib.colors.LogNorm(),
#                    )
#                    axs_samp[i,j].set_title('samp observable')
#                    axs_samp[i,j].set_xlabel(self.dataset.observable_names[i+1])
#                    axs_samp[i,j].set_ylabel(self.dataset.observable_names[j])
#                    fig_samp.colorbar(H[3],ax=axs_samp[i,j])
#
#        plt.show()


    @torch.no_grad()
    def sample(self,idx,n_samples):
        if isinstance(idx,int):
            idx = [idx]
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx)
        cond = self.dataset.conditions[idx]
        cond = self.cond_prep(cond.to(self.device))
        with torch.no_grad():
            if cond.shape[-1] == 0:
                samp = self.flow.sample(
                    n_samples = cond.shape[0]*n_samples,
                    context = cond,
                )
            else:
                samp = self.flow.sample(
                    n_samples = n_samples,
                    context = cond,
                ).permute(1,0,2)
        samp = self.obs_prep.inverse(samp).cpu()
        return samp


    def plot_sampling(self,idx,n_samples,bins=50,show_directions=False):
        # Get true values #
        ri = self.dataset.ri[idx]
        di = self.dataset.di[idx]
        true_rf = self.dataset.rf[idx]
        true_df = self.dataset.df[idx]

        # Sample #
        samp = self.sample(idx,n_samples)[:,0,:]
        a1,a2,s1,s2 = samp.unbind(dim=-1)

        samp_rf,samp_df = observables_to_positions(
            a1,a2,s1,s2,
            ri.unsqueeze(0),
            di.unsqueeze(0),
            self.dataset.low,self.dataset.high,
        )
        samp_rf = samp_rf.numpy()
        samp_df = samp_df.numpy()

        # test to see if the forward+inverse are fine #
        test_obs = self.dataset.observables[idx:idx+1]
        test_rf,test_df = observables_to_positions(
            *test_obs.unbind(dim=-1),
            ri.unsqueeze(0),
            di.unsqueeze(0),
            self.dataset.low,self.dataset.high,
        )
        assert (true_rf-test_rf).sum() < 1e-6
        assert (true_df-test_df).sum() < 1e-6

        fig, ax = plot_3d_hist2d_faces(
            x = samp_rf[:,0],
            y = samp_rf[:,1],
            z = samp_rf[:,2],
            xlim = (self.dataset.low[0].numpy(),self.dataset.high[0].numpy()),
            ylim = (self.dataset.low[1].numpy(),self.dataset.high[1].numpy()),
            zlim = (self.dataset.low[2].numpy(),self.dataset.high[2].numpy()),
            bins = bins,
        )
        plt.suptitle(f'Event {idx} : P = {self.dataset.pi[idx].item():.2f} GeV',fontsize=16)
        # Add true points #
        ax.scatter(
            ri[0], ri[1], ri[2],
            color='forestgreen',
            s=300,
            marker='o',
            edgecolors='black',
            linewidths=2.0,
            label='Initial position',
            depthshade=False,
            alpha=0.6,
            zorder=100,
        )
        ax.scatter(
            true_rf[0], true_rf[1], true_rf[2],
            color='darkred',
            s=300,
            marker='o',
            edgecolors='black',
            linewidths=2.0,
            label='True final position',
            depthshade=False,
            alpha=0.6,
            zorder=100,
        )
        # plot directions
        di = di / di.norm()
        L = 2.0
        start = ri - L * di
        ax.quiver(
            start[0], start[1], start[2],
            di[0], di[1], di[2],
            length=L,
            normalize=True,
            color='forestgreen',
            linewidth=3,
            arrow_length_ratio=0.25,
        )
        ax.quiver(
            true_rf[0], true_rf[1], true_rf[2],
            true_df[0], true_df[1], true_df[2],
            length=L,
            normalize=True,
            color='darkred',
            linewidth=3,
            arrow_length_ratio=0.25,
        )
        if show_directions:
            ax.quiver(
                samp_rf[:,0], samp_rf[:,1], samp_rf[:,2],
                samp_df[:,0], samp_df[:,1], samp_df[:,2],
                length=L/2,
                normalize=True,
                color='grey',
                linewidth=1,
                arrow_length_ratio=0.15,
            )
        return fig

