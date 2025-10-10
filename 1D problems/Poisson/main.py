import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PoissonNeuralNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=3, hidden_dim=50, activation=nn.Tanh, device="cpu"):
        super().__init__()

        self.activation = activation
        self.device = torch.device(device)
        self.mse = nn.MSELoss()

        layers = [nn.Linear(input_dim, hidden_dim), self.activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), self.activation()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers).to(self.device)
        self.init_weights()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_history = {"pde": [], "bc": [], "total": []}

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
    def gradients(self, y, x):
        return torch.autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
    
    def pde_residual(self, x):
        u = self.forward(x)

        u_x = self.gradients(u, x)
        u_xx = self.gradients(u_x, x)

        f = torch.sin(x)
        return u_xx + f

    def loss_function(self, x_f, x_bnd, u_bnd):
        res = self.pde_residual(x_f)
        mse_pde = torch.mean(res**2)

        u_pred_bnd = self.forward(x_bnd)
        mse_bnd = self.mse(u_pred_bnd, u_bnd)

        loss = mse_pde + mse_bnd
        return loss, mse_pde.item(), mse_bnd.item()


    def train(self, x_f, x_bnd, u_bnd, epochs=1000):
        x_f = torch.tensor(x_f, dtype=torch.float32, requires_grad=True, device=self.device)
        x_bnd = torch.tensor(x_bnd, dtype=torch.float32, requires_grad=True, device=self.device)
        u_bnd = torch.tensor(u_bnd, dtype=torch.float32, requires_grad=True, device=self.device)

        self.start_time = perf_counter()

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            loss, mse_pde, mse_bnd = self.loss_function(x_f, x_bnd, u_bnd)
            loss.backward()
            self.optimizer.step()

            self.loss_history['pde'].append(mse_pde)
            self.loss_history['bc'].append(mse_bnd)
            self.loss_history['total'].append(loss.item())

            if epoch % 100 == 0 or epoch == epochs - 1:
                elapsed = perf_counter() - self.start_time

                print(f"[Adam] Epoch {epoch}, "
                      f"Loss = {loss.item():.6e}, "
                      f"pde = {self.loss_history['pde'][-1]:.3e}, "
                      f"bc = {self.loss_history['bc'][-1]:.3e}, "
                      f"Time = {elapsed:.2f} s")

    
    def predict(self, x_star):
        x_t = torch.tensor(x_star, dtype=torch.float32, device=self.device).reshape(-1, 1)

        with torch.no_grad():
            u = self.forward(x_t).detach().cpu().numpy().ravel()

        return u


def analytical_solution(x):
    return np.sin(x) - (1 + np.sin(1)) * x + 1





if __name__ == "__main__":
    set_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoissonNeuralNet(device=device)

    x_bounds = [0.0, 1.0]
    x_b = np.array([0.0, 1.0]).reshape(-1, 1) # x=0, x=1
    u_b = np.array([1.0, 0.0]).reshape(-1, 1) # u(x=0)=1, u(x=1)=0

    N = 200 # collocation points
    x_f = np.random.uniform(low=x_bounds[0], high=x_bounds[1], size=N).reshape(-1, 1)


    model.train(x_f, x_b, u_b, epochs=1000)
    
    # saving model weights
    torch.save(model.state_dict(), "trained_model.pt")

    x_test = np.linspace(0, 1, N)
    u_true = analytical_solution(x_test)
    u_pred = model.predict(x_test)

    print(u_true.shape, u_pred.shape)

    rel_l2 = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
    print(f"Relative L2 error: {rel_l2:.3e}")

    # post process
    figure, axis = plt.subplots(1, 2, figsize=(14, 5))
    
    axis[0].plot(x_test, u_true, label="Exact solution")
    axis[0].plot(x_test, u_pred, linestyle='--', label="PINN prediction")
    axis[0].scatter(x_b, u_b, marker='o')  # boundary points
    axis[0].set_xlabel("x")
    axis[0].set_ylabel("P(x)")
    axis[0].set_title("1D Poisson equation")
    axis[0].legend()
    axis[0].grid(True)

    pde_loss = model.loss_history["pde"]
    bc_loss = model.loss_history["bc"]
    loss = model.loss_history["total"]

    axis[1].plot(pde_loss, label="pde")
    axis[1].plot(bc_loss, label="bc")
    axis[1].plot(loss, label="total")
    axis[1].set_yscale("log")
    axis[1].set_xlabel("Epoch")
    axis[1].set_ylabel("Loss")
    axis[1].set_title("Loss in log scale")
    axis[1].legend()
    axis[1].grid(True)
    
    figure.tight_layout()
    plt.show()


