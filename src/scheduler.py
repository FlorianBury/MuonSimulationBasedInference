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
        mapper_losses = [],
        n_samples = 1,
        batch_size = 1000,
        scheduler_params = None,
    ):
        self.data_dataset = dataset
        self.mapper = mapper
        self.propagator = propagator
        self.callbacks = callbacks
        self.mapper_losses = mapper_losses
        self.n_samples = n_samples
        self.batch_size = batch_size

        self.propagator.init_box(self.data_dataset)
        self.device = device
        if self.device is None:
            self.device = 'cpu'
        self.mapper.set_to_device(self.device)

        if scheduler_params is None:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.mapper.optimizer,**scheduler_params)


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
                    idx = idx,
                )
        else:
            rf, df, pf = self.propagator.propagate(
                mapper = self.mapper,
                ri = ri,
                di = di,
                pi = pi,
                idx = idx,
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
            print(f"Iteration {it+1}/{n_iterations} [{(it+1)/n_iterations*100:5.2f}]% - LR = {self.mapper.lr:.3e}")
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
            losses = self._loss(it,sim_datasets,idx)
            losses.update(
                {
                    mapper_loss.name : mapper_loss(self.mapper)
                    for mapper_loss in self.mapper_losses
                }
            )
            loss_tot = sum(losses.values())
            s = f'Loss : {loss_tot:7.5f} ='
            for name,loss in losses.items():
                s += f' + {loss:7.5f} ({name})'
            print (s)
            for callback in self.callbacks:
                callback.on_losses_end(
                    iteration = it,
                    lr = self.mapper.lr,
                    losses = {
                        'Total ' : loss_tot.item(),
                        **{
                            name : loss.item()
                            for name,loss in losses.items()
                        },
                    },
                )
            # Perform gradient step #
            checker = ModelParameterChecker(self.mapper)
            self.mapper.train()
            self.mapper.optimize(loss_tot)
            print(f"Voxel Min: {self.mapper.voxels.min().item():.4f} | Max: {self.mapper.voxels.max().item():.4f}")
            if checker.has_changed():
                print (checker.compare())
            if self.scheduler is not None:
                self.scheduler.step(loss_tot)
            # Callbacks #
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
                sim_dataset._observables[sim_dataset.mask]
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

        att, rep = energy_distance_loss(
            x = true_obs,
            y = samp_obs,
            gamma = self.gamma,
            p = self.p,
        )
        losses = {
            'Attraction' : att.mean(),
            'Repulsion' : rep.mean(),
        }
        return losses


class GaussianLikelihoodScheduler(BaseScheduler):
    """Maximum-likelihood mapper optimisation on the data only (no simulator samples)."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.step_size = self.propagator.step_size
        self._obs = None


    def optimize_step(self, it):
        idx = torch.randperm(len(self.data_dataset))[: self.batch_size]
        mask = getattr(self.data_dataset, "mask", None)
        if mask is not None:
            idx = idx[mask[idx]]
        ri, rf, pi, obs = [
            t[idx].to(self.device)
            for t in [
                self.data_dataset.ri,
                self.data_dataset.rf,
                self.data_dataset.pi,
                self.data_dataset._observables,
            ]
        ]

        loss, q = gaussian_nll(self.mapper, ri, rf , pi, obs, self.step_size, jitter=True)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite likelihood: check lambda >= 0 and the track lengths")

        pull = q.mean().item() / 2 # ~1 when the model is calibrated
        #print(f"NLL = {loss.item():9.5f}   <q>/2 = {pull:6.3f}   ({q.shape[0]} muons)")
        return loss

    def optimize(self, n_iterations):
        for callback in self.callbacks:
            callback.on_optimisation_start(self.mapper)
        for it in range(n_iterations):
            print(f"Iteration {it+1}/{n_iterations} [{(it+1)/n_iterations*100:5.2f}]% - LR = {self.mapper.lr:.3e}")
            loss = self.optimize_step(it)
            extra_losses = {
                mapper_loss.name : mapper_loss(self.mapper)
                for mapper_loss in self.mapper_losses
            }
            loss_tot = loss + sum(extra_losses.values())
            s = f'Loss : {loss_tot:.3f} = {loss:.3f} (Mapping)'
            for name,extra_loss in extra_losses.items():
                s += f' + {extra_loss:.3f} ({name})'
            print (s)
            self.mapper.train()
            self.mapper.optimize(loss)
            if self.scheduler is not None:
                self.scheduler.step(loss_tot)
            for callback in self.callbacks:
                callback.on_losses_end(
                    iteration = it,
                    lr = self.mapper.lr,
                    losses = {
                        'Loss' : loss_tot.item(),
                        'Mapping' : loss.item(),
                        **{
                            name : extra_loss.item()
                            for name,extra_loss in extra_losses.items()
                        }
                    },
                )
            for callback in self.callbacks:
                callback.on_iteration_end(
                    iteration = it,
                    mapper = self.mapper,
                )

        for callback in self.callbacks:
            callback.on_optimisation_end(self.mapper)


