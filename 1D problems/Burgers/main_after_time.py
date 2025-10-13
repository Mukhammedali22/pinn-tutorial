import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import perf_counter

# u_t + u*u_x = nu*u_xx

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def Analytical_solution(x, t, nu=0.05):
    return 0.5 - 0.5 * np.tanh((x - 0.5 * t) / (4 * nu))


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=4, hidden_dim=32, activation=nn.Tanh):
        super().__init__()
       
        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class BurgersPinn(nn.Module):
    def __init__(self, model, lb, ub, nu=0.05, w_f=1.0, w_bc=1.0, w_ic=1.0, device="cpu"):
        super().__init__()

        self.nu = nu # viscosity coefficient
        self.w_f = w_f
        self.w_bc = w_bc
        self.w_ic = w_ic

        self.lb = lb # lower bound [x_min, t_min]
        self.ub = ub # upper bound [x_max, t_max]

        self.model = model
        self.mse = nn.MSELoss()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-8,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        self.lbfgs_step = 0
        self.loss_history = {"pde": [], "bc": [], "ic": [], "total": []}

    def gradients(self, y, x):
        return torch.autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
    
    def to_tensor(self, x_np, data_type=torch.float32, requires_grad=False):
        return torch.tensor(x_np, dtype=data_type, requires_grad=requires_grad, device=self.device)
        
    def pde_loss(self):
        u = self.model(torch.hstack([self.x_f, self.t_f]))

        u_t = self.gradients(u, self.t_f)
        u_x = self.gradients(u, self.x_f)
        u_xx = self.gradients(u_x, self.x_f)

        loss = torch.mean((u_t + u * u_x - self.nu * u_xx)**2)
        return loss
    
    def bc_loss(self):
        u_pred = self.model(torch.hstack([self.x_bnd, self.t_bnd]))

        u_pred_x = self.gradients(u_pred, self.x_bnd)
        # u(x=L,t)=1, u_x(x=R,t)=0
        left_side = (self.x_bnd == self.lb[0])

        left_loss = self.mse(u_pred[left_side], self.u_bnd[left_side])
        right_loss = self.mse(u_pred_x[~left_side], self.u_bnd[~left_side])   
        loss = left_loss + right_loss
        return loss

    def ic_loss(self):
        u_pred_ic = self.model(torch.hstack([self.x_ic, self.t_ic]))
        # loss = self.mse(u_pred_ic, self.u_ic)
        loss = torch.mean((u_pred_ic - self.u_ic)**2)
        return loss

    def loss_function(self):
        mse_pde = self.pde_loss()
        mse_bnd = self.bc_loss()
        mse_ic = self.ic_loss()
        
        loss = self.w_f * mse_pde + self.w_bc * mse_bnd + self.w_ic * mse_ic
        return loss, mse_pde.item(), mse_bnd.item(), mse_ic.item()


    def train(self, X_f, X_bnd, u_bnd, X_ic, u_ic, epochs=1000):
        self.x_f = self.to_tensor(X_f[:, 0:1], requires_grad=True)
        self.t_f = self.to_tensor(X_f[:, 1:2], requires_grad=True)

        self.x_bnd = self.to_tensor(X_bnd[:, 0:1], requires_grad=True)
        self.t_bnd = self.to_tensor(X_bnd[:, 1:2], requires_grad=True)

        self.x_ic = self.to_tensor(X_ic[:, 0:1], requires_grad=True)
        self.t_ic = self.to_tensor(X_ic[:, 1:2], requires_grad=True)

        self.u_bnd = self.to_tensor(u_bnd)
        self.u_ic = self.to_tensor(u_ic)

        self.start_time = perf_counter()

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            loss, mse_pde, mse_bnd, mse_ic = self.loss_function()
            loss.backward()
            self.optimizer.step()

            self.loss_history['pde'].append(mse_pde)
            self.loss_history['bc'].append(mse_bnd)
            self.loss_history['ic'].append(mse_ic)
            self.loss_history['total'].append(loss.item())

            if epoch % 100 == 0 or epoch == epochs - 1:
                elapsed = perf_counter() - self.start_time

                print(f"[Adam] Epoch {epoch}/{epochs}, "
                      f"Loss = {loss.item():.6e}, "
                      f"pde = {mse_pde:.3e}, "
                      f"ic = {mse_ic:.3e}, "
                      f"bc = {mse_bnd:.3e}, "
                      f"Time = {elapsed:.2f} s")

        loss = self.optimizer_lbfgs.step(self.closure)
        elapsed = perf_counter() - self.start_time

        print(f"[LBFGS] Optimization finished, final values\n"
              f"[LBFGS] Iter {self.lbfgs_step}, "
              f"Loss = {self.loss_history['total'][-1]:.6e}, "
              f"pde = {self.loss_history['pde'][-1]:.3e}, "
              f"ic = {self.loss_history['ic'][-1]:.3e}, "
              f"bc = {self.loss_history['bc'][-1]:.3e}, "
              f"Time = {elapsed:.2f} s")


    def closure(self):
        self.optimizer_lbfgs.zero_grad()
        loss, mse_pde, mse_bnd, mse_ic = self.loss_function()
        loss.backward()

        self.loss_history['pde'].append(mse_pde)
        self.loss_history['bc'].append(mse_bnd)
        self.loss_history['ic'].append(mse_ic)
        self.loss_history['total'].append(loss.item())

        if self.lbfgs_step % 100 == 0:
            elapsed = perf_counter() - self.start_time

            print(f"[LBFGS] Iter {self.lbfgs_step}, "
                    f"Loss = {loss.item():.6e}, "
                    f"pde = {mse_pde:.3e}, "
                    f"ic = {mse_ic:.3e}, "
                    f"bc = {mse_bnd:.3e}, "
                    f"Time = {elapsed:.2f} s")

        self.lbfgs_step += 1
        return loss


    def predict(self, x, t):
        x = self.to_tensor(x).reshape(-1, 1)
        t = self.to_tensor(t).reshape(-1, 1)

        with torch.no_grad():
            u = self.model(torch.hstack([x, t])).detach().cpu().numpy().reshape(-1, 1)

        return u


if __name__ == "__main__":
    set_seed(1234)

    # data preparation
    x_bounds = (-1.0, 3.0)
    t_bounds = (0.0, 4.0)

    lb = [x_bounds[0], t_bounds[0]]
    ub = [x_bounds[1], t_bounds[1]]

    N = 4000 # collocation points
    N_b = 200 # boundary points for each side
    N_i = 1000 # initial points

    t_b = np.random.uniform(low=t_bounds[0], high=t_bounds[1], size=(N_b, 1))
    left_side = np.hstack([lb[0] * np.ones_like(t_b), t_b])
    right_side = np.hstack([ub[0] * np.ones_like(t_b), t_b])
    X_b = np.vstack([left_side, right_side])
    u_b = (X_b[:, 0:1] == lb[0]) # u=1 at x=-1, u=0 at x=3

    x = np.random.uniform(low=x_bounds[0], high=x_bounds[1], size=(N_i, 1))
    X_i = np.hstack([x, np.zeros_like(x)])
    u_i = Analytical_solution(x, t=0)

    X_f = np.random.uniform(low=lb, high=ub, size=(N, 2))

    # setup pinn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_dim=2, output_dim=1, hidden_layers=4, hidden_dim=50)
    pinn = BurgersPinn(model, lb, ub, w_f=1.0, w_bc=1.0, w_ic=1.0, device=device)

    pinn.train(X_f, X_b, u_b, X_i, u_i, epochs=2000)
    
    # saving model weights
    torch.save(pinn.state_dict(), "trained_model_2.pt")

    # try to predict for t = [4.0, 8.0]
    x_test = np.linspace(lb[0], ub[0], 200)
    t_test = np.linspace(4.0, 8.0, 200)
    X, T = np.meshgrid(x_test, t_test)

    u_pred = pinn.predict(X, T).reshape(X.shape)
    u_true = Analytical_solution(X, T)
    error = np.abs(u_pred - u_true)

    print(u_true.shape, u_pred.shape)

    rel_l2 = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {rel_l2:.3e}")


    # post process
    plt.figure(figsize=(7,5))
    plt.contourf(x_test, t_test, u_pred, levels=100, cmap='turbo')
    plt.colorbar(label='U(t,x)')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Predicted solution")
    plt.tight_layout()
    plt.savefig('Figure 1.2.png')
    plt.show()


    plt.figure(figsize=(7,5))
    times = [4.0, 6.0, 8.0]
    for t_val in times:
        idx = (np.abs(t_test - t_val)).argmin()
        plt.plot(x_test, u_true[idx, :], label=f"Exact t={t_val}", lw=2)
        plt.plot(x_test, u_pred[idx, :], '--', label=f"PINN t={t_val}")
    plt.xlabel("x")
    plt.ylabel("U(t,x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figure 2.2.png')
    plt.show()


    plt.figure(figsize=(7,5))
    plt.contourf(x_test, t_test, error, levels=100, cmap='inferno')
    plt.colorbar(label='|Error|')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Absolute error")
    plt.tight_layout()
    plt.savefig('Figure 3.2.png')
    plt.show()


    pde_loss = pinn.loss_history["pde"]
    bc_loss = pinn.loss_history["bc"]
    loss = pinn.loss_history["total"]

    plt.figure(figsize=(7, 5))
    plt.plot(pde_loss, label="pde")
    plt.plot(bc_loss, label="bc")
    plt.plot(loss, label="total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss value in log scale")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Figure 4.2.png')
    plt.show()


    def make_gif(X, T, U_pred, U_true, filename="pinn_exact_solution.gif"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1, ax2 = axes

        y_min = min(U_pred.min(), U_true.min()) - 0.1
        y_max = max(U_pred.max(), U_true.max()) + 0.1
        t_values = T[:, 0]

        line1, = ax1.plot([], [], lw=2, color='darkorange')
        ax1.set_xlim(X.min(), X.max())
        ax1.set_ylim(y_min, y_max)
        ax1.set_title("PINN prediction")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x,t)")
        ax1.grid(True)

        line2, = ax2.plot([], [], lw=2, color='blue')
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(y_min, y_max)
        ax2.set_title("Exact solution")
        ax2.set_xlabel("x")
        ax2.grid(True)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return (line1, line2)

        def animate(i):
            line1.set_data(X[i, :], U_pred[i, :])
            line2.set_data(X[i, :], U_true[i, :])
            fig.suptitle(f"t = {t_values[i]:.2f}")
            return (line1, line2)

        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(t_values), interval=80, blit=True)

        ani.save(filename, writer="pillow", fps=15)
        plt.close(fig)
        print(f"GIF saved as '{filename}'")


    make_gif(X, T, u_pred, u_true, filename="pinn_solution_2.gif")
