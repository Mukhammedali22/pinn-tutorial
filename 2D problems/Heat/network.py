import torch
import torch.nn as nn
from time import perf_counter


class PINN(nn.Module):
    def __init__(self, lb, ub, k=1.0, layers=[2, 50, 50, 50, 50, 1], 
                 w_f=1.0, w_ic=1.0, w_bc=1.0, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.lb = self.to_tensor(lb)
        self.ub = self.to_tensor(ub)
        # constants
        self.k = k
        self.w_f = w_f
        self.w_ic = w_ic
        self.w_bc = w_bc

        self.net = self.build_network(layers).to(self.device)
        self.init_weights()

        self.lbfgs_step = 0
        self.loss_history = {"pde": [], "bc": [], "ic": [], "total": []}

    def build_network(self, layers):
        modules = []
        for i in range(len(layers)-2):
            modules += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        modules += [nn.Linear(layers[-2], layers[-1])]
        return nn.Sequential(*modules)

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.net.parameters(), lr=1.0,
            max_iter=50000, tolerance_grad=1e-8,
            tolerance_change=1e-9, line_search_fn="strong_wolfe"
        )

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def gradients(self, y, x):
        return torch.autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
    
    def to_tensor(self, x_np, data_type=torch.float32, requires_grad=False):
        return torch.tensor(x_np, dtype=data_type, requires_grad=requires_grad, device=self.device)
        
    def pde_residual(self):
        u = self.forward(torch.hstack([self.x_f, self.y_f, self.t_f]))

        u_t = self.gradients(u, self.t_f)
        u_x = self.gradients(u, self.x_f)
        u_xx = self.gradients(u_x, self.x_f)
        u_y = self.gradients(u, self.y_f)
        u_yy = self.gradients(u_y, self.y_f)
        
        return u_t - self.k * (u_xx + u_yy)

    def pde_loss(self):
        res = self.pde_residual()
        return torch.mean(res**2)
    
    def bc_loss(self):
        u_pred = self.forward(torch.hstack([self.x_bnd, self.y_bnd, self.t_bnd]))
        return torch.mean((u_pred - self.u_bnd)**2)
    
    def ic_loss(self):
        u_pred = self.forward(torch.hstack([self.x_ic, self.y_ic, self.t_ic]))
        return torch.mean((u_pred - self.u_ic)**2)

    def loss_function(self):
        mse_pde = self.pde_loss()
        mse_bnd = self.bc_loss()
        mse_ic = self.ic_loss()
        
        loss = self.w_f * mse_pde + self.w_bc * mse_bnd + self.w_ic * mse_ic
        return loss, mse_pde.item(), mse_bnd.item(), mse_ic.item()

    def train(self, X_f, X_bnd, u_bnd, X_ic, u_ic, lr=1e-3, epochs=1000, lbfgs=True):
        self.x_f = self.to_tensor(X_f[:, 0:1], requires_grad=True)
        self.y_f = self.to_tensor(X_f[:, 1:2], requires_grad=True)
        self.t_f = self.to_tensor(X_f[:, 2:3], requires_grad=True)

        self.x_bnd = self.to_tensor(X_bnd[:, 0:1], requires_grad=True)
        self.y_bnd = self.to_tensor(X_bnd[:, 1:2], requires_grad=True)
        self.t_bnd = self.to_tensor(X_bnd[:, 2:3], requires_grad=True)
        self.u_bnd = self.to_tensor(u_bnd)

        self.x_ic = self.to_tensor(X_ic[:, 0:1])
        self.y_ic = self.to_tensor(X_ic[:, 1:2])
        self.t_ic = self.to_tensor(X_ic[:, 2:3])
        self.u_ic = self.to_tensor(u_ic)

        self.configure_optimizers(lr=lr)

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

        if lbfgs:
            loss = self.optimizer_lbfgs.step(self.closure)
            elapsed = perf_counter() - self.start_time
            self.end_time = elapsed

            print(f"[LBFGS] Optimization finished, final values\n"
                f"[LBFGS] Iter {self.lbfgs_step}, "
                f"Loss = {self.loss_history['total'][-1]:.6e}, "
                f"pde = {self.loss_history['pde'][-1]:.3e}, "
                f"ic = {self.loss_history['ic'][-1]:.3e}, "
                f"bc = {self.loss_history['bc'][-1]:.3e}, "
                f"Time = {self.end_time:.2f} s")


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


    def predict(self, x, y, t):
        x = self.to_tensor(x).reshape(-1, 1)
        y = self.to_tensor(y).reshape(-1, 1)
        t = self.to_tensor(t).reshape(-1, 1)

        with torch.no_grad():
            u = self.net(torch.hstack([x, y, t])).detach().cpu().numpy().reshape(-1, 1)

        return u

