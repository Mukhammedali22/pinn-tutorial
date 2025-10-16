import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import qmc 

from network import PINN


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(1234)

    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)
    t_bounds = (0.0, 1.0)

    lb = np.array([x_bounds[0], y_bounds[0], t_bounds[0]])
    ub = np.array([x_bounds[1], y_bounds[1], t_bounds[1]])

    N = 5000 # collocation points
    N_b = 1000 # boundary points for each side
    N_i = 1000 # initial points

    sampler = lambda dim: qmc.LatinHypercube(d=dim)
    X_f = sampler(lb.shape[0]).random(N)
    X_f = qmc.scale(X_f, lb, ub)

    X_i = sampler(lb.shape[0]).random(N_i)
    X_i = qmc.scale(X_i, lb, ub)
    X_i[:, -1] = 0.0
    u_i = np.zeros_like(X_i[:, 0:1])

    x_left = np.hstack([lb[0]*np.ones((N_b, 1)), 
                        sampler(1).random(N_b),
                        sampler(1).random(N_b)])

    x_right = np.hstack([ub[0]*np.ones((N_b, 1)), 
                        sampler(1).random(N_b),
                        sampler(1).random(N_b)])

    x_bottom = np.hstack([sampler(1).random(N_b), 
                        lb[1]*np.ones((N_b, 1)),
                        sampler(1).random(N_b)])

    x_top = np.hstack([sampler(1).random(N_b), 
                        ub[1]*np.ones((N_b, 1)),
                        sampler(1).random(N_b)])

    X_b = np.vstack([x_left, x_right, x_bottom, x_top])
    X_b = qmc.scale(X_b, lb, ub)
    u_b = np.zeros_like(X_b[:, 0:1])

    u_b[(X_b[:, 0] == 0.0) & (X_b[:, 1] < 0.7)] = 0.0
    u_b[(X_b[:, 0] == 0.0) & (X_b[:, 1] >= 0.7)] = 1.0
    u_b[(X_b[:, 0] == 1.0) & (X_b[:, 1] < 0.3)] = 1.0
    u_b[(X_b[:, 0] == 1.0) & (X_b[:, 1] >= 0.3)] = 0.0
    u_b[(X_b[:, 1] == 0.0) | (X_b[:, 1] == 1.0)] = 0.0


    print(X_f.shape, X_b.shape, u_b.shape, X_i.shape, u_i.shape)
    layers = [3, 50, 50, 50, 50, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn = PINN(lb, ub, layers=layers, 
                w_f=1.0, w_ic=10.0, w_bc=100.0, device=device)

    pinn.train(X_f, X_b, u_b, X_i, u_i, lr=1e-3, epochs=5000)
    pinn.save_model("heat_pinn_model.pt")

    # pinn.load_model("heat_pinn_model.pt")

    nx, ny = (200, 200)
    x_test = np.linspace(lb[0], ub[0], nx)
    y_test = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(x_test, y_test)
    T = np.ones_like(X)

    u_pred = pinn.predict(x=X, y=Y, t=T).reshape(X.shape)
    np.savetxt(f"pinn_sol.txt", u_pred, fmt="%.6f", delimiter="\t")

    # post process
    plt.figure(figsize=(7,5))
    plt.contourf(X, Y, u_pred, levels=100, cmap='turbo')
    plt.colorbar(label='U(t,x,y)')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Predicted solution at t=1.0")
    plt.tight_layout()
    plt.savefig('Figure 1.png')
    plt.show()

    nt = 41
    t_values = np.linspace(t_bounds[0], t_bounds[1], nt)

    fig, ax = plt.subplots(figsize=(7, 5))

    cax = ax.contourf(X, Y, np.zeros_like(X), levels=100, cmap='turbo')
    cb = fig.colorbar(cax, ax=ax, label="U(t,x,y)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("PINN Predicted Solution")

    def update(frame):
        t = t_values[frame]
        T = np.full_like(X, t)
        u_pred = pinn.predict(x=X, y=Y, t=T).reshape(X.shape)

        ax.clear()
        cax = ax.contourf(X, Y, u_pred, levels=100, cmap='turbo')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Predicted solution at t={t:.2f}")
        cb.ax.clear()
        fig.colorbar(cax, ax=ax, cax=cb.ax)

    ani = animation.FuncAnimation(fig, update, frames=nt, interval=500, blit=False)

    ani.save("pinn_solution.gif", writer="pillow", fps=10)

    plt.close(fig)

    pde_loss = pinn.loss_history["pde"]
    ic_loss = pinn.loss_history["ic"]
    bc_loss = pinn.loss_history["bc"]
    loss = pinn.loss_history["total"]

    plt.figure(figsize=(7, 5))
    plt.plot(pde_loss, label="pde")
    plt.plot(bc_loss, label="bc")
    plt.plot(ic_loss, label="ic")
    plt.plot(loss, label="total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss value in log scale")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figure 2.png')
    plt.show() 