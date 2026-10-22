import math
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt

from tools import *
from surrogates import *
from mappers import *
from dataset import *


class BaseScheduler:
    def __init__(
        self,
        dataset,
        mapper,
        propagator,
        device = None,
        callbacks = [],
        n_samples = 1,
        batch_size = 1000,
    ):
        self.data_dataset = dataset
        self.mapper = mapper
        self.propagator = propagator
        self.callbacks = callbacks
        self.n_samples = n_samples
        self.batch_size = batch_size

        self.propagator.init_box(self.data_dataset)
        self.device = device
        if self.device is None:
            self.device = 'cpu'

        self.mapper.set_to_device(self.device)

    def pre_train_mapper(self,sampling_function,loss_cutoff,batch_size,max_epochs=math.inf):
        loss = 1e10
        epoch = 0
        while loss > loss_cutoff:
            x,t = sampling_function(batch_size)
            x = x.to(self.device)
            t = t.to(self.device)
            loss = self.mapper.loss(x,t)
            self.mapper.optimize(loss)
            epoch += 1
            if epoch % 1000 == 0:
                print (f'Epoch {epoch} : loss = {loss:1.3e}')
            if epoch > max_epochs:
                break
        print (f'Pre-training stopped after {epoch} epochs : loss = {loss:1.3e}')

    def train_data_surrogate(self):
        self.train_surrogate(
            model = self.data_surrogate,
            dataset = self.data_dataset,
        )

    def train_surrogate(self,model,dataset):
        model.init_dataset(dataset)
        model.set_gradients(True)
        model.train()

        n_epochs = [10,5,3]
        batch_size = 5000
        lrs = [1e-3,1e-4,1e-5]
        for epochs, lr in zip(n_epochs,lrs):
            model.train_model(
                epochs = epochs,
                batch_size = batch_size,
                lr = lr,
            )

        model.set_gradients(False)
        model.eval()

    def propagate(self,batch_size,no_grad=True):
        print (f'Propagating batch of {batch_size}x{self.n_samples} for mapper update')
        # select batch of muons #
        idx = torch.randperm(len(self.data_dataset))[:batch_size]
        ri = self.data_dataset.ri[idx]
        di = self.data_dataset.di[idx]
        pi = self.data_dataset.pi[idx]

        ri = ri.unsqueeze(1).expand(-1, self.n_samples, -1).reshape(batch_size*self.n_samples,-1)
        di = di.unsqueeze(1).expand(-1, self.n_samples, -1).reshape(batch_size*self.n_samples,-1)
        pi = pi.unsqueeze(1).expand(-1, self.n_samples, -1).reshape(batch_size*self.n_samples,-1)

        # Propagate to positions #
        if no_grad:
            with torch.no_grad():
                rf, df, pf = self.propagator.propagate(
                    mapper = self.mapper,
                    ri = ri,
                    di = di,
                    pi = pi,
                )
        else:
            rf, df, pf = self.propagator.propagate(
                mapper = self.mapper,
                ri = ri,
                di = di,
                pi = pi,
            )
        ri = ri.reshape(batch_size,self.n_samples,-1)
        di = di.reshape(batch_size,self.n_samples,-1)
        pi = pi.reshape(batch_size,self.n_samples,-1)
        rf = rf.reshape(batch_size,self.n_samples,-1)
        df = df.reshape(batch_size,self.n_samples,-1)

        sim_datasets = [
            MuonDataset(
                ri = ri[:,i],
                rf = rf[:,i],
                di = di[:,i],
                df = df[:,i],
                pi = pi[:,i],
                selection = self.data_dataset.selection,
            )
            for i in range(self.n_samples)
        ]
        mask = torch.stack(
            [
                sim_dataset.mask
                for sim_dataset in sim_datasets
            ]
        ).all(dim=0)
        for sim_dataset in sim_datasets:
            sim_dataset.mask = mask

        idx = idx[mask]
        return sim_datasets,idx

    def optimize(
        self,
        n_iterations,
    ):
        self._init()
        for callback in self.callbacks:
            callback.on_optimisation_start(self.mapper)
        for it in range(n_iterations):
            print (f'Iteration {it+1}/{n_iterations} [{(it+1)/n_iterations*100:5.2f}]%')
            # Pre-step #
            self._pre_step()
            # Propagate muons with gradient #
            sim_datasets,idx = self.propagate(self.batch_size,no_grad=False)
            for callback in self.callbacks:
                callback.on_observables_end(
                    iteration = it,
                    obs_samp = sim_datasets[0].observables.detach(),
                    obs_true = self.data_dataset.observables.detach(),
                    obs_names = self.data_dataset.observable_names,
                )
            # Calculate loss #
            loss = self._loss(it,sim_datasets,idx)
            print (f'Loss = {loss:.5f}')
            # Perform gradient step #
            self.mapper.train()
            self.mapper.optimize(loss)
            for callback in self.callbacks:
                callback.on_iteration_end(
                    iteration = it,
                    mapper = self.mapper,
                )

        for callback in self.callbacks:
            callback.on_optimisation_end(self.mapper)

    #def run(
    #    self,
    #    n_iterations,
    #    batch_size,
    #):
    #    self.initialize()
    #    self.optimize(n_iterations,batch_size)



class TwoFlowScheduler(BaseScheduler):
    def __init__(
        self,
        sim_surrogate,
        data_surrogate,
        n_simulations,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sim_surrogate = sim_surrogate
        self.data_surrogate = data_surrogate
        self.n_simulations = n_simulations
        self.sim_surrogate.set_to_device(self.device)
        self.data_surrogate.set_to_device(self.device)


    def initialize(
        self,
        n_epochs,
        batch_sizes,
        lrs,
    ):
        self.pre_train_mapper()

        self.train_loss_surrogate(n_epochs,batch_sizes,lrs)
        fig = self.data_surrogate.validate_model(batch_size=5000)
        plt.show()

    def _init(self):
        self.data_surrogate.set_gradients(False)
        self.data_surrogate.eval()
        self.log_prob_final = self.data_surrogate.log_prob_dataset(
            dataset = self.data_surrogate.valid_dataset,
            batch_size = 5000,
        )

    def _pre_step(self):
        print ('Training simulation surrogate')
        self.train_surrogate(
            model = self.sim_surrogate,
            dataset = sim_dataset,
        )
        fig = self.sim_surrogate.validate_model(batch_size=5000)
        plt.show()
        # Propagate muons with gradient #
        print (f'Propagating batch of {n_gradients} for mapper update')
        sim_dataset,_ = self.propagate(n_gradients,no_grad=False)
        for callback in self.callbacks:
            callback.on_observables_end(
                iteration = it,
                obs_samp = sim_dataset.observables.detach(),
                obs_true = self.data_dataset.observables.detach(),
                obs_names = self.data_dataset.observable_names,
            )

    def _loss(self,it,sim_dataset,idx):
        # Total loss #
        print ('Computing losses')
        log_prob_sim = self.sim_surrogate.log_prob(
            obs = sim_dataset.observables,
            cond = sim_dataset.conditions,
        )
        log_prob_data = self.data_surrogate.log_prob(
            obs = sim_dataset.observables,
            cond = sim_dataset.conditions,
        )
        coverage = self.sim_surrogate.log_prob_dataset(
            dataset = self.data_surrogate.valid_dataset,
            batch_size = 5000,
        )
        for callback in self.callbacks:
            callback.on_surrogate_end(it,log_prob_sim.detach().cpu(),log_prob_data.detach().cpu())
        loss = (log_prob_sim-log_prob_data).mean()
        print (f'Loss = {log_prob_sim.mean().item():8.5f} (sim) - {log_prob_data.mean().item():8.5f} (data) = {loss.item():8.5f} (total)')
        for callback in self.callbacks:
            callback.on_losses_end(
                iteration = it,
                losses = {
                    r'Reverse $KL(p_{\psi}||q_{\phi})$' : loss.item(),
                    r'Forward $KL(q_{\phi}||p_{\psi})$' : (log_prob_final-coverage).mean().item(),
                    r'$E[\log q_{\phi}(x_{\theta})]$' : log_prob_data.mean().item(),
                    r'$E[\log p_{\psi}(x_{\theta})]$' : log_prob_sim.mean().item(),
                    r'$E[\log q_{\phi}(x_{data})]$' : log_prob_final.mean().item(),
                    r'$E[\log p_{\psi}(x_{data})]$' : coverage.mean().item(),
                },
            )
        return loss



class EnergyDistanceScheduler(BaseScheduler):
    def __init__(
        self,
        data_surrogate = None,
        gamma = 1.,
        p = 2.,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_surrogate = data_surrogate
        if self.data_surrogate is not None:
            self.data_surrogate.set_to_device(self.device)
        assert self.n_samples >= 2
        self.gamma = gamma
        self.p = p

    def initialize(
        self,
        n_epochs,
        batch_sizes,
        lrs,
    ):
        self.pre_train_mapper()

        if self.data_surrogate is not None:
            self.train_loss_surrogate(n_epochs,batch_sizes,lrs)
            fig = self.data_surrogate.validate_model(batch_size=5000)
            plt.show()

    def _init(self):
        if self.data_surrogate is not None:
            self.data_surrogate.set_gradients(False)
            self.data_surrogate.eval()

    def _pre_step(self):
        pass

    def _loss(self,it,sim_datasets,idx):
        print ('Computing losses')
        # Get propagated observables #
        samp_obs = torch.stack(
            [
                sim_dataset.observables
                for sim_dataset in sim_datasets
            ],
            dim = 1,
        )
        # Get true observables #
        if self.data_surrogate is not None:
            assert self.n_samples is not None
            true_obs = self.data_surrogate.sample(
                idx = idx,
                n_samples = self.n_samples,
            )
        else:
            true_obs = self.data_dataset.observables[idx].unsqueeze(1)

        # Weight #
        norms = torch.norm(samp_obs, dim=-1)
        weights = 1.0 + self.gamma * (norms ** self.p) # Shape: (N, M)
        # Attractive : samp vs true observables #
        eps = 1e-8
        diff_sq = ((samp_obs[:, :, None, :] - true_obs[:, None, :, :]) ** 2).sum(-1)
        weighted_diff_sq = weights[:, None, :] * diff_sq
        att = torch.sqrt(weighted_diff_sq + eps).mean((1, 2))
        # Repulsive : samp vs samp observables #
        rep = torch.sqrt(((samp_obs[:, :, None] - samp_obs[:, None]) ** 2).sum(-1) + eps)
        K = self.n_samples
        off = ~torch.eye(K, dtype=torch.bool, device=samp_obs.device)
        rep = rep[:, off].sum(-1) / (K * (K - 1))
        # Total loss #
        tot_loss = (att - 0.5 * rep).mean()
        # Callbacks #
        for callback in self.callbacks:
            callback.on_losses_end(
                iteration = it,
                losses = {
                    'Attraction' : att.mean().item(),
                    'Repulsion' : rep.mean().item(),
                    'Loss' : tot_loss.item(),
                },
            )
        return tot_loss






#        log_prob_sim = self.sim_surrogate.log_prob(
#            obs = sim_dataset.observables,
#            cond = sim_dataset.conditions,
#        )
#        log_prob_data = self.data_surrogate.log_prob(
#            obs = sim_dataset.observables,
#            cond = sim_dataset.conditions,
#        )
#        coverage = self.sim_surrogate.log_prob_dataset(
#            dataset = self.data_surrogate.valid_dataset,
#            batch_size = 5000,
#        )
#        for callback in self.callbacks:
#            callback.on_surrogate_end(it,log_prob_sim.detach().cpu(),log_prob_data.detach().cpu())
#        loss = (log_prob_sim-log_prob_data).mean()
#        print (f'Loss = {log_prob_sim.mean().item():8.5f} (sim) - {log_prob_data.mean().item():8.5f} (data) = {loss.item():8.5f} (total)')
#        for callback in self.callbacks:
#            callback.on_losses_end(
#                iteration = it,
#                losses = {
#                    r'Reverse $KL(p_{\psi}||q_{\phi})$' : loss.item(),
#                    r'Forward $KL(q_{\phi}||p_{\psi})$' : (log_prob_final-coverage).mean().item(),
#                    r'$E[\log q_{\phi}(x_{\theta})]$' : log_prob_data.mean().item(),
#                    r'$E[\log p_{\psi}(x_{\theta})]$' : log_prob_sim.mean().item(),
#                    r'$E[\log q_{\phi}(x_{data})]$' : log_prob_final.mean().item(),
#                    r'$E[\log p_{\psi}(x_{data})]$' : coverage.mean().item(),
#                },
#            )
#        return loss
