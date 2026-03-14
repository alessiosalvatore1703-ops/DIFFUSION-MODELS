import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from .distributions import gmm_pdf



def animate_points(points_history, means, covs, weights):
    x = np.linspace(-3, 1, 200)
    y = np.linspace(-2, 2, 200)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    Z = gmm_pdf(grid, means, covs, weights).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.contourf(xx, yy, Z, levels=40, cmap="Reds")
    scat = ax.scatter(points_history[0][:,0], points_history[0][:,1], c="blue", s=40)
    ax.set_xlim(-3,1); ax.set_ylim(-2,2); ax.set_aspect("equal")

    def update(i):
        scat.set_offsets(points_history[i])
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=len(points_history), interval=100, blit=True)

    plt.close(fig) 
    return ani

def plot_gaussians(means, covs, weights):
    x = np.linspace(-3, 1, 200)
    y = np.linspace(-2, 2, 200)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    Z = gmm_pdf(grid, means, covs, weights).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.contourf(xx, yy, Z, levels=40, cmap="Reds")
    plt.show()