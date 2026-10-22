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

    def on_losses_end(self, iteration: int, losses: dict):
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
        return iteration % self.frequency == 0

    def plot(self,iteration: int) -> None:
        pass

class LossPlotCallback(Callback):
    def __init__(self,log=False,**kwargs):
        self.log = log
        super().__init__(**kwargs)

    def on_optimisation_start(self,mapper):
        self.iterations = []
        self.losses = {}

    def on_optimisation_end(self,mapper):
        self.plot()

    def on_losses_end(self, iteration: int, losses: dict):
        self.iterations.append(iteration)
        for key,val in losses.items():
            if key not in self.losses.keys():
                self.losses[key] = [val]
            else:
                self.losses[key].append(val)
        self.plot(iteration)

    def plot(self,iteration=None):
        if iteration is None or self.should_run(iteration):
            fig = plt.figure()
            for name,values in self.losses.items():
                plt.plot(
                    self.iterations,
                    values,
                    label = name,
                )
            plt.xlabel('Iterations')
            plt.legend()
            if self.log:
                plt.yscale('log')
            plt.show()

class MapperPlotCallback(Callback):
    def __init__(self,low,high,**kwargs):
        super().__init__(**kwargs)
        self.low = low
        self.high = high

    def on_optimisation_end(self,mapper):
        self.plot(None,mapper)

    def on_iteration_end(self, iteration: int, mapper):
        self.plot(iteration,mapper)

    def plot(self,iteration,mapper):
        if iteration is None or self.should_run(iteration):
            for value in [1.0,3.5,6.0]:
                fig = mapper.plot_slice(
                    direction = 'y',
                    value = value,
                    low = [self.low[0],self.low[2]],
                    high = [self.high[0],self.high[2]],
                    batch_size = 5000,
                    points = 50,
                    show_materials = True,
                )
            for value in [1.0,2.0,3.0]:
                fig = mapper.plot_slice(
                    direction = 'z',
                    value = value,
                    low = [self.low[0],self.low[1]],
                    high = [self.high[0],self.high[1]],
                    batch_size = 5000,
                    points = 50,
                    show_materials = True,
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


