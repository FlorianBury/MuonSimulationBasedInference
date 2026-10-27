import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC

class Callback(ABC):
    def __init__(self, frequency: int = 1):
        if frequency < 1:
            raise ValueError("frequency must be >= 1")

        self.frequency = frequency

    def on_optimisation_start(self,mapper):
        """Called once when optimisation starts."""
        pass

    def on_optimisation_end(self,mapper):
        """Called once when optimisation ends."""
        pass

    def on_iteration_start(self, iteration: int, mapper):
        """Called at the start of an iteration, respecting frequency."""
        pass

    def on_iteration_end(self, iteration: int, mapper):
        """Called at the end of an iteration, respecting frequency."""
        pass

    def on_propagate_end(self, iteration: int, ri, di, pi, rf, df, pf):
        """Called at the end of an iteration, respecting frequency."""
        pass

    def on_losses_end(self, iteration: int, losses: dict, lr: float):
        """Called at the end of an iteration, respecting frequency."""
        pass

    def on_observables_end(self, iteration: int, obs_samp, obs_true, obs_names):
        """Called at the end of an iteration, respecting frequency."""
        pass

    def on_surrogate_end(self, iteration: int, log_prob_samp, log_prob_true):
        """Called at the end of an iteration, respecting frequency."""
        pass

    def should_run(self, iteration: int) -> bool:
        """Determine whether the callback should run at this iteration."""
        return iteration > 0 and iteration % self.frequency == 0

    def plot(self,iteration: int) -> None:
        pass

class LossPlotCallback(Callback):
    def __init__(self,scale='linear',**kwargs):
        self.scale = scale
        super().__init__(**kwargs)

    def on_optimisation_start(self,mapper):
        self.iterations = []
        self.lrs = []
        self.losses = {}

    def on_optimisation_end(self,mapper):
        self.plot()

    def on_losses_end(self, iteration: int, losses: dict, lr: float):
        self.iterations.append(iteration)
        self.lrs.append(lr)
        for key,val in losses.items():
            if key not in self.losses.keys():
                self.losses[key] = [val]
            else:
                self.losses[key].append(val)
        self.plot(iteration)

    def plot(self,iteration=None):
        if iteration is None or self.should_run(iteration):
            fig,ax1 = plt.subplots()
            ax2 = ax1.twinx()
            for name,values in self.losses.items():
                ax1.plot(
                    self.iterations,
                    values,
                    label = name,
                )
            ax2.plot(
                self.iterations,
                self.lrs,
                linestyle = 'dashed',
                color = 'black',
            )
            ax1.set_xlabel('Iterations',fontsize=14)
            ax1.set_ylabel('Losses',fontsize=14)
            ax2.set_ylabel('LR',fontsize=14)
            ax1.set_yscale(self.scale) #,linthresh=1)
            ax2.set_yscale('log')
            ax1.legend()
            plt.show()

class MapperPlotCallback(Callback):
    def __init__(self,low,high,points=50,xs=[],ys=[],zs=[],log=False,**kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.points = points
        self.log = log

    def on_optimisation_end(self,mapper):
        self.plot(None,mapper)

    def on_iteration_end(self, iteration: int, mapper):
        self.plot(iteration,mapper)

    def plot(self,iteration,mapper):
        if iteration is None or self.should_run(iteration):
            for x in self.xs:
                fig = mapper.plot_slice(
                    direction = 'x',
                    value = x,
                    low = [self.low[1],self.low[2]],
                    high = [self.high[1],self.high[2]],
                    batch_size = 5000,
                    points = self.points,
                    show_materials = False,
                    log = self.log,
                )
            for y in self.ys:
                fig = mapper.plot_slice(
                    direction = 'y',
                    value = y,
                    low = [self.low[0],self.low[2]],
                    high = [self.high[0],self.high[2]],
                    batch_size = 5000,
                    points = self.points,
                    show_materials = False,
                    log = self.log,
                )
            for z in self.zs:
                fig = mapper.plot_slice(
                    direction = 'z',
                    value = z,
                    low = [self.low[0],self.low[1]],
                    high = [self.high[0],self.high[1]],
                    batch_size = 5000,
                    points = self.points,
                    show_materials = False,
                    log = self.log,
                )
            plt.show()

class ObservablesPlotCallback(Callback):
    def on_observables_end(self, iteration: int, obs_samp, obs_true, obs_names):
        if iteration is None or self.should_run(iteration):
            fig,axs = plt.subplots(
                ncols = obs_true.shape[1],
                nrows = 1,
                figsize=(5*obs_true.shape[1],4),
            )
            plt.subplots_adjust(wspace=0.4)
            for i in range(obs_true.shape[1]):
                bins = np.linspace(
                    min(obs_true[:,i].min(),obs_samp[:,i].min()),
                    max(obs_true[:,i].max(),obs_samp[:,i].max()),
                    50,
                )
                axs[i].hist(
                    obs_true[:,i].numpy(),
                    bins = bins,
                    density = True,
                    histtype = 'step',
                    label = 'True observable',
                    color = 'royalblue',
                )
                axs[i].hist(
                    obs_samp[:,i].numpy(),
                    bins = bins,
                    density = True,
                    histtype = 'step',
                    label = 'Propagated observable',
                    color = 'orange',
                )
                axs[i].legend(fontsize=10)
                axs[i].set_xlabel(obs_names[i],fontsize=14)
                axs[i].set_yscale('log')
            plt.tight_layout()
            plt.show()
            return fig

class LogProbPlotCallback(Callback):
    def on_surrogate_end(self, iteration: int, log_prob_sim, log_prob_data):
        if iteration is None or self.should_run(iteration):
            fig = plt.figure(figsize=(6,5))
            bins = np.linspace(
                min(log_prob_sim.min(),log_prob_data.min()),
                max(log_prob_sim.max(),log_prob_data.max()),
                50,
            )
            H = plt.hist2d(
                log_prob_sim,
                log_prob_data,
                bins = bins,
                norm = matplotlib.colors.LogNorm(),
                cmin = 1,
            )
            plt.plot(
                [bins[0],bins[0]],
                [bins[-1],bins[-1]],
                linestyle = 'dashed',
                color = 'red',
            )
            plt.xlabel(r'True $\log P$')
            plt.ylabel(r'Propagation $\log P$')
            plt.colorbar(H[3])
            plt.tight_layout()
            plt.show()
            return fig


