import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import perf_counter
from scipy.stats import qmc 

# 2D Laplace equation
# u_xx + u_yy = 0

# Boundary conditions
# P(x=0, 0<y<0.7) = 0
# P(x=0, 0.7<y<1) = 1
# P(x=1, 0<y<0.3) = 1
# P(x=1, 0.3<y<1) = 0
# P(x, y=0) = 0
# P(x, y=1) = 0

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def Jacobi_method(N, M, eps=1e-6, stop_iteration=3e4) -> np.ndarray:
    """Jacobi method for solving 2D Laplace equation"""
    P_old = np.zeros((M, N))
    P_new = np.zeros_like(P_old)

    dx = 1.0 / (N - 1)
    dy = 1.0 / (M - 1)

    M1 = int(0.3 * M)
    M2 = int(0.7 * M)

    P_old[0:M2, 0] = 0
    P_old[M1:M, N-1] = 0
    P_old[0, 0:N] = 0
    P_old[M-1, 0:N] = 0 
    P_old[M2:M, 0] = 1
    P_old[0:M1, N-1] = 1

    iteration = 0
    maximum = 1
    while maximum > eps and iteration < stop_iteration:
        P_new[1:M-1, 1:N-1] = (
            dy**2*(P_old[1:M-1, 2:N] + P_old[1:M-1, 0:N-2]) 
            + dx**2*(P_old[2:M, 1:N-1] + P_old[0:M-2, 1:N-1])
            ) / (2*(dx**2 + dy**2))
        
        P_new[0:M2, 0] = 0
        P_new[M1:M, N-1] = 0
        P_new[0, 0:N] = 0
        P_new[M-1, 0:N] = 0 
        P_new[M2:M, 0] = 1
        P_new[0:M1, N-1] = 1

        maximum = np.max(np.abs(P_new - P_old))
        # print(f"{iteration = }\t{maximum = }")
        P_old = P_new.copy()
        iteration += 1

    print(f"Number of iterations: {iteration}")
    print(f"Maximum absolute difference: {maximum}")
    
    return P_new


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


class LaplacePinn(nn.Module):
    def __init__(self, model, w_f=1.0, w_bc=1.0, device="cpu"):
        super().__init__()

        self.w_f = w_f
        self.w_bc = w_bc

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
        u = self.model(torch.hstack([self.x_f, self.y_f]))

        u_x = self.gradients(u, self.x_f)
        u_xx = self.gradients(u_x, self.x_f)
        u_y = self.gradients(u, self.y_f)
        u_yy = self.gradients(u_y, self.y_f)

        loss = torch.mean((u_xx + u_yy)**2)

        return loss
    
    def bc_loss(self):
        u_pred = self.model(torch.hstack([self.x_bnd, self.y_bnd]))
        loss = torch.mean((u_pred - self.u_bnd)**2)

        return loss

    def loss_function(self):
        mse_pde = self.pde_loss()
        mse_bnd = self.bc_loss()
        loss = self.w_f * mse_pde + self.w_bc * mse_bnd

        return loss, mse_pde.item(), mse_bnd.item()

    def train(self, X_f, X_bnd, u_bnd, epochs=1000, lbgfs=True):
        self.x_f = self.to_tensor(X_f[:, 0:1], requires_grad=True)
        self.y_f = self.to_tensor(X_f[:, 1:2], requires_grad=True)

        self.x_bnd = self.to_tensor(X_bnd[:, 0:1], requires_grad=True)
        self.y_bnd = self.to_tensor(X_bnd[:, 1:2], requires_grad=True)
        self.u_bnd = self.to_tensor(u_bnd)

        self.start_time = perf_counter()
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            loss, mse_pde, mse_bnd = self.loss_function()
            loss.backward()
            self.optimizer.step()

            self.loss_history['pde'].append(mse_pde)
            self.loss_history['bc'].append(mse_bnd)
            self.loss_history['total'].append(loss.item())

            if epoch % 100 == 0 or epoch == epochs - 1:
                elapsed = perf_counter() - self.start_time

                print(f"[Adam] Epoch {epoch}/{epochs}, "
                      f"Loss = {loss.item():.6e}, "
                      f"pde = {mse_pde:.3e}, "
                      f"bc = {mse_bnd:.3e}, "
                      f"Time = {elapsed:.2f} s")

        if lbgfs:
            loss = self.optimizer_lbfgs.step(self.closure)
            elapsed = perf_counter() - self.start_time
            self.end_time = elapsed

            print(f"[LBFGS] Optimization finished, final values\n"
                f"[LBFGS] Iter {self.lbfgs_step}, "
                f"Loss = {self.loss_history['total'][-1]:.6e}, "
                f"pde = {self.loss_history['pde'][-1]:.3e}, "
                f"bc = {self.loss_history['bc'][-1]:.3e}, "
                f"Time = {self.end_time:.2f} s")


    def closure(self):
        self.optimizer_lbfgs.zero_grad()
        loss, mse_pde, mse_bnd = self.loss_function()
        loss.backward()

        self.loss_history['pde'].append(mse_pde)
        self.loss_history['bc'].append(mse_bnd)
        self.loss_history['total'].append(loss.item())

        if self.lbfgs_step % 100 == 0:
            elapsed = perf_counter() - self.start_time

            print(f"[LBFGS] Iter {self.lbfgs_step}, "
                    f"Loss = {loss.item():.6e}, "
                    f"pde = {mse_pde:.3e}, "
                    f"bc = {mse_bnd:.3e}, "
                    f"Time = {elapsed:.2f} s")

        self.lbfgs_step += 1
        return loss


    def predict(self, x, y):
        x = self.to_tensor(x).reshape(-1, 1)
        y = self.to_tensor(y).reshape(-1, 1)

        with torch.no_grad():
            u = self.model(torch.hstack([x, y])).detach().cpu().numpy().reshape(-1, 1)

        return u


if __name__ == "__main__":
    set_seed(1234)

    # data preparation
    x_bounds = (0.0, 1.0)
    y_bounds = (0.0, 1.0)

    lb = [x_bounds[0], y_bounds[0]]
    ub = [x_bounds[1], y_bounds[1]]

    N = 5000 # collocation points
    N_b = 200 # boundary points for each side

    sampler = qmc.LatinHypercube(d=2)
    X_f = sampler.random(N)

    sampler = qmc.LatinHypercube(d=1)
    left_side = np.hstack([x_bounds[0]*np.ones((N_b, 1)), sampler.random(N_b)])
    right_side = np.hstack([x_bounds[1]*np.ones((N_b, 1)), sampler.random(N_b)])
    bottom_side = np.hstack([sampler.random(N_b), y_bounds[0]*np.ones((N_b, 1))])
    top_side = np.hstack([sampler.random(N_b), y_bounds[1]*np.ones((N_b, 1))])
    X_b = np.vstack([left_side, right_side, bottom_side, top_side])

    u_b = np.zeros((X_b.shape[0], 1))

    u_b[(X_b[:, 0] == 0.0) & (X_b[:, 1] < 0.7)] = 0.0
    u_b[(X_b[:, 0] == 0.0) & (X_b[:, 1] >= 0.7)] = 1.0
    u_b[(X_b[:, 0] == 1.0) & (X_b[:, 1] < 0.3)] = 1.0
    u_b[(X_b[:, 0] == 1.0) & (X_b[:, 1] >= 0.3)] = 0.0
    u_b[(X_b[:, 1] == 0.0) | (X_b[:, 1] == 1.0)] = 0.0

    print(X_f.shape, X_b.shape, u_b.shape)

    # setup pinn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuralNet(input_dim=2, output_dim=1, hidden_layers=6, hidden_dim=50)
    pinn = LaplacePinn(model, w_f=1.0, w_bc=100.0, device=device)

    pinn.train(X_f, X_b, u_b, epochs=5000, lbgfs=True)
    
    # saving model weights
    torch.save(pinn.state_dict(), "trained_model.pt")

    nx, ny = (200, 200)
    x_test = np.linspace(lb[0], ub[0], nx)
    y_test = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(x_test, y_test)

    u_pred = pinn.predict(X, Y).reshape(X.shape)

    u_num = Jacobi_method(nx, ny)
    error = np.abs(u_pred - u_num)
    print(f"Error {error.max():.3f}, {np.mean(error):.3f}")

    np.savetxt(f"num_sol.txt", u_num, fmt="%.6f", delimiter="\t")
    np.savetxt(f"pinn_sol.txt", u_pred, fmt="%.6f", delimiter="\t")

    print(u_num.shape, u_pred.shape)
    print("u_num", u_num.min(), u_num.max(), np.abs(u_num).max())
    print("u_pred", u_pred.min(), u_pred.max(), np.abs(u_pred).max())

    rel_l2 = np.linalg.norm(u_num - u_pred) / np.linalg.norm(u_num)
    print(f"Relative L2 error: {rel_l2:.3e}")


    # post process
    plt.figure(figsize=(7,5))
    plt.contourf(x_test, y_test, u_pred, levels=100, cmap='turbo')
    plt.colorbar(label='U(t,x)')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted solution")
    plt.tight_layout()
    plt.savefig('Figure 1.png')
    plt.show()

    plt.figure(figsize=(7,5))
    plt.contourf(x_test, y_test, error, levels=100, cmap='inferno')
    plt.colorbar(label='|Error|')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.tight_layout()
    plt.savefig('Figure 2.png')
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
    plt.savefig('Figure 3.png')
    plt.show()
