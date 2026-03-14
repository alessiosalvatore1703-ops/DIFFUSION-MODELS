# DIFFUSION-MODELS
Project for AML class at ETHZ

## Stochastic Deep Learning & Generative Modeling
This repository contains a collection of research and implementation notebooks focused on the intersection of stochastic processes and deep generative models. The progression covers numerical simulations of SDEs, sampling from complex probability distributions, and training diffusion models.

### Repository Structure
The core logic is divided into three major components:

#### 1. Numerical Methods for SDEs (euler_maruyama.ipynb)
This notebook serves as the mathematical foundation for the repository. It implements the Euler-Maruyama method, a numerical technique used to simulate paths of Stochastic Differential Equations.

Key Concepts: Drift and Diffusion coefficients, Wiener processes (Brownian motion), and Itô isometry.

Implementations: Simulation of the Ornstein-Uhlenbeck process and analysis of time-discretized stochastic increments.

#### 2. Energy-Based Sampling (GMM_langevinSampling.ipynb)
Explores how to draw samples from a complex target distribution (e.g., a Mixture of Gaussians) using the Score Function and Langevin Dynamics.

Key Concepts: Log-density gradients, "responsibilities" in Gaussian Mixture Models (GMM), and the Langevin MCMC algorithm.

Visualizations: Animations showing random particles converging to the modes of a distribution by following the gradient of the log-probability.

#### 3. Generative Diffusion Models (ddpm.ipynb)
A complete implementation of Denoising Diffusion Probabilistic Models (DDPM) applied to the MNIST dataset.
