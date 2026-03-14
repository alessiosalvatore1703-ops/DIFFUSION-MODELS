import numpy as np
import math
from itertools import product
from typing import Callable, Optional, Union
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# SIZES = {"big":(10, 7, 15, 19)}

@dataclass
class DataPlotter:
    euler_maruyama_function: Callable 
    drift: Callable
    diffusion: Callable
    mean: Optional[Callable] = None
    std: Optional[Callable] = None
    t0: float = 0.0
    x0: Union[float, list] = 0.0
    T: float = 2.0
    dt: Union[float, list] = 0.01
    npaths: Union[int, list] = 10
    paths: tuple[np.array, np.matrix] = None
    mean_path: np.matrix = None

    @staticmethod
    def listify(val):
        return val if isinstance(val, list) else [val]
        
    def grid(self):
        drift_list = DataPlotter.listify(self.drift)
        diffusion_list = DataPlotter.listify(self.diffusion)
        mean_list = DataPlotter.listify(self.mean)
        std_list = DataPlotter.listify(self.std)
        t0_list = DataPlotter.listify(self.t0)
        x0_list = DataPlotter.listify(self.x0)
        T_list = DataPlotter.listify(self.T)
        dt_list = DataPlotter.listify(self.dt)
        npaths_list = DataPlotter.listify(self.npaths)
        
        grid = []
        for params in product(drift_list, diffusion_list, mean_list, std_list, t0_list, x0_list, dt_list, T_list, npaths_list):
            drift_fn, diffusion_fn, mean_fn, std_fn, t0_val, x0_val, dt_val, T_val, npaths_val = params

            # Create a copy of DataPlotter for this combination
            dp = DataPlotter(
                drift=drift_fn,
                diffusion=diffusion_fn,
                mean=mean_fn,
                std=std_fn,
                t0=t0_val,
                x0=x0_val,
                T=T_val,
                dt=dt_val,
                npaths=npaths_val,
                euler_maruyama_function = self.euler_maruyama_function
            )

            # Simulate paths
            dp.paths = self.euler_maruyama_function(f=dp.drift, g=dp.diffusion, t0=dp.t0, x0=dp.x0, T=dp.T, dt=dp.dt, npaths=dp.npaths)
            if mean_fn is not None:
                ts = np.linspace(dp.t0, dp.T, 2000)
                dp.mean_path = (ts, mean_fn(dp.x0, dp.t0, ts))
            
            grid.append(dp)

        return grid



def plot_paths_of_euler_maruyama_method(grid, n_repetitions=1):

    nrows = n_repetitions
    ncols = len(grid)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), sharex=True, sharey=True)
    if len(grid) > 1:  axes = axes.flatten()
    else: axes = [axes]

    for i in range(nrows):
        for j,dp in enumerate(grid):
            ax = axes[i*ncols+j]
            
            
            ts, paths = dp.paths
            for xs in paths:
                ax.plot(ts, xs, '-', alpha=0.5)

            if dp.mean_path is not None:
                ts_mean, mean = dp.mean_path
                ax.plot(ts_mean, mean, lw=2, label='Expected mean', color='black', alpha=.5)

            ax.set_title(rf"Euler-Maruyama Sample Paths $(\Delta t = {dp.dt})$")
            ax.set_xlabel('$T$')
            ax.set_ylabel('$X(T)$')
            ax.grid(alpha=0.3)
            ax.legend()


def nicely_plot_estimated_distribution_euler_maruyama_method(grid, bins=30):

    nrows=2
    ncols = len(grid)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), sharex='row', sharey='row')
    axes = axes.flatten()

    for i, dp in enumerate(grid):
        ax = axes[i]

        if dp.mean_path is not None:
            ts_mean, mean = dp.mean_path
            ax.plot(ts_mean, mean, lw=2, label='Expected mean', color='black')
            
        ts, paths = dp.paths
        
        mean = paths.mean(axis=0)
        q01 = np.percentile(paths, 1, axis=0)
        q99 = np.percentile(paths, 99, axis=0)
        q10 = np.percentile(paths, 10, axis=0)
        q90 = np.percentile(paths, 90, axis=0)
        q25 = np.percentile(paths, 25, axis=0)
        q75 = np.percentile(paths, 75, axis=0)

        
        for j in range(min(20,dp.npaths)):
            ax.plot(ts, paths[j,:], linewidth=0.8, alpha=0.15)

        ax.fill_between(ts, q01, q99, alpha=0.25, label='01-99% band')
        ax.fill_between(ts, q10, q90, alpha=0.25, label='10-90% band')
        ax.fill_between(ts, q25, q75, alpha=0.4, label='25-75% band')


        ax.plot(ts, mean, lw=2.2, label='Empirical mean', zorder=10)
        ax.plot(ts, paths[0], lw=1.5, label='Example path', alpha=0.6)


        ax.set_xlabel('$T$')
        ax.set_ylabel('$X(T)$')
        ax.set_title(rf'SDE empirical distribution $(\Delta t = {dp.dt})$')
        ax.legend()
        ax.grid(alpha=0.3)
        
        

        ft = min(ts[-1], dp.T)
        endpoints = paths[:, -1]
        kde = gaussian_kde(endpoints)
        xs = np.linspace(endpoints.min() - 1e-1, endpoints.max() + 1e-1, 200)

        ax = axes[ncols + i]
        ax.hist(endpoints, bins=bins, density=True, alpha=0.4, label=f'Histogram of $X({ts[-1]:.3g})$')
        ax.plot(xs, kde(xs), lw=2)
        ax.axvline(np.mean(endpoints), color='k', linestyle='--', label='Empirical mean')
        
        if dp.mean is not None:
            mu = dp.mean(dp.x0, dp.t0, ft)
            ax.axvline(mu, linestyle='--', label='Mean')

            if dp.std is not None:
                sigma = dp.std(dp.t0, ft)
                
                gauss = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-1/(2*sigma**2) * (xs-mu)**2) 
                ax.plot(xs, gauss, linestyle='--', label='Expected Distribution')


        ax.set_xlabel('$X(T)$')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution at $X({ft:.3g})$')
        ax.legend()


def nicely_plot_ornstein_uhlenbeck(grid):
    plt.figure(figsize=(8.5, 5.5))
    ax = plt.gca()

    for i,(dp,theta,sigma,mu,euler_maruyama_ft) in enumerate(grid):
        ts, xs = euler_maruyama_ft(dp.drift, dp.diffusion, dp.t0, dp.x0, dp.dt, dp.T, dp.npaths)
        label = r"$X_0={x0},\quad  \mu={mu},\quad  \theta={theta},\quad  \sigma={sigma}$".format(x0=f"{dp.x0:.3g}", mu=f"{mu:.3g}", theta=f"{theta:.3g}", sigma=f"{sigma:.3g}")
        color = f"C{i}"
        
        for x in xs:
            ax.plot(ts, x, label=label, color=color, linewidth=1.5, alpha=0.85)
            label = ""

        mean_analytic = dp.mean(dp.x0, dp.t0, ts)
        ax.plot(ts, mean_analytic, color="0.5", linestyle="-", linewidth=2.0, alpha=0.5)

    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$X$', fontsize=12)
    ax.set_title('Different sample paths of different OU-processes ', fontsize=12)
    ax.grid(True, which='both', linestyle='-', linewidth=0.7, alpha=0.7)
    leg = ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()


def nicely_plot_ornstein_uhlenbeck_distribution(grid, bins=20):
    ncols, nrows = len(grid), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), sharex=None, sharey=None)
    axes = axes.flatten()

    for i, (dp, theta, sigma, mu, euler_maruyama_ft) in enumerate(grid):
        ax = axes[i]
        ts, xs = euler_maruyama_ft(dp.drift, dp.diffusion, dp.t0, dp.x0, dp.dt, dp.T, dp.npaths)
        xs = np.asarray(xs)
        label = r"$X_0={x0},\quad  \mu={mu},\quad  \theta={theta},\quad  \sigma={sigma}$".format(x0=f"{dp.x0:.3g}", mu=f"{mu:.3g}", theta=f"{theta:.3g}", sigma=f"{sigma:.3g}")

        n_paths = xs.shape[0]
        for j in range(min(20, n_paths)):
            ax.plot(ts, xs[j, :], linewidth=0.8, alpha=.2)

        mean = xs.mean(axis=0)
        p01 = np.percentile(xs, 1, axis=0)
        p10 = np.percentile(xs, 10, axis=0)
        p25 = np.percentile(xs, 25, axis=0)
        p75 = np.percentile(xs, 75, axis=0)
        p90 = np.percentile(xs, 90, axis=0)
        p99 = np.percentile(xs, 99, axis=0)

        ax.fill_between(ts, p01, p99, alpha=0.18, label="01-99% band")
        ax.fill_between(ts, p10, p90, alpha=0.22, label="10-90% band")
        ax.fill_between(ts, p25, p75, alpha=0.30, label="25-75% band")

        ax.plot(ts, mean, lw=2.2, label="ensemble mean", alpha=0.9)

        mean_analytic = dp.mean(dp.x0, dp.t0, ts)
        ax.plot(ts, mean_analytic, color="0.5", linestyle="-", linewidth=2.0, alpha=0.9, label="Analytic mean")

        if n_paths >= 1:
            ax.plot(ts, xs[0, :], lw=1.6, label="Example path", alpha=0.7, color='C5')
            
        ax.set_xlabel(r'$t$', fontsize=12)
        ax.set_ylabel(r'$X$', fontsize=12)
        ax.set_title('Path distribution of different OU-processes', fontsize=12)
        ax.grid(True, which='both', linestyle='-', linewidth=0.7, alpha=0.7)
        ax.legend(fontsize=9)


        ft = min(ts[-1], dp.T)
        endpoints = xs[:, -1]
        kde = gaussian_kde(endpoints)
        xs = np.linspace(endpoints.min() - 1e-1, endpoints.max() + 1e-1, 200)

        ax = axes[ncols + i]
        ax.hist(endpoints, bins=bins, density=True, alpha=0.4, label=f'Histogram of $x({ts[-1]:.3g})$')
        ax.plot(xs, kde(xs), lw=2)
        ax.axvline(np.mean(endpoints), color='k', linestyle='--', label='Empirical mean')
        
        if dp.mean is not None:
            mu = dp.mean(dp.x0, dp.t0, ft)
            ax.axvline(mu, linestyle='--', label='Mean')

            if dp.std is not None:
                std = dp.std(dp.t0, ft)
                
                gauss = 1/(np.sqrt(2 * np.pi) * std) * np.exp(-1/(2*std**2) * (xs-mu)**2) 
                ax.plot(xs, gauss, linestyle='--', label='Expected Distribution')


        ax.set_xlabel('$X(T)$')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution at $X({ft:.3g})$')
        ax.legend()