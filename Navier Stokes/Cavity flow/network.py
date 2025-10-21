import torch
import torch.nn as nn
import numpy as np
from time import perf_counter


class PINN(nn.Module):
    def __init__(self, lb, ub, nu=0.01, rho=1.0, layers=[3, 50, 50, 50, 50, 2], 
                 w_f=1.0, w_ic=1.0, w_bc=1.0, device="cpu"):
        super().__init__()

        self.device = torch.device(device)
        self.lb = self.to_tensor(lb, data_type=torch.float64)
        self.ub = self.to_tensor(ub, data_type=torch.float64)

        # constants
        self.nu = nu # viscosity
        self.rho = rho # density
        self.w_f = w_f
        self.w_ic = w_ic
        self.w_bc = w_bc

        self.mse = nn.MSELoss()
        self.net = self.build_network(layers).to(self.device).double()
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
            tolerance_change=1e-12, 
            line_search_fn="strong_wolfe"
        )

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def gradients(self, y, x, create_graph=True, retain_graph=True):
        return torch.autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=create_graph, retain_graph=retain_graph
        )[0]
    
    def to_tensor(self, x_np, data_type=torch.float64, requires_grad=False):
        return torch.tensor(x_np, dtype=data_type, requires_grad=requires_grad, device=self.device)

    def set_inputs(self, X, data_type=torch.float64, requires_grad=False):
        x = self.to_tensor(X[:, 0:1], data_type=data_type, requires_grad=requires_grad)
        y = self.to_tensor(X[:, 1:2], data_type=data_type, requires_grad=requires_grad)
        t = self.to_tensor(X[:, 2:3], data_type=data_type, requires_grad=requires_grad)
        return x, y, t
     
    def pde_residual(self):
        res = self.forward(torch.hstack([self.x_f, self.y_f, self.t_f]))
        psi, p = res[:, 0:1], res[:, 1:2]

        u = self.gradients(psi, self.y_f)
        v = -self.gradients(psi, self.x_f)

        u_t = self.gradients(u, self.t_f)
        u_x = self.gradients(u, self.x_f)
        u_xx = self.gradients(u_x, self.x_f)
        u_y = self.gradients(u, self.y_f)
        u_yy = self.gradients(u_y, self.y_f)

        v_t = self.gradients(v, self.t_f)
        v_x = self.gradients(v, self.x_f)
        v_xx = self.gradients(v_x, self.x_f)
        v_y = self.gradients(v, self.y_f)
        v_yy = self.gradients(v_y, self.y_f)

        p_x = self.gradients(p, self.t_f)
        p_y = self.gradients(p, self.y_f)

        f = u_t + u*u_x + v*u_y + p_x/self.rho - self.nu*(u_xx + u_yy)
        g = v_t + u*v_x + v*v_y + p_y/self.rho - self.nu*(v_xx + v_yy)
        return f, g

    def pde_loss(self):
        f, g = self.pde_residual()
        mse_f = torch.mean(f**2) 
        mse_g = torch.mean(g**2)
        return mse_f + mse_g
    
    def bc_loss(self):
        res = self.forward(torch.hstack([self.x_bnd, self.y_bnd, self.t_bnd]))
        psi, p = res[:, 0:1], res[:, 1:2]
        u_pred = self.gradients(psi, self.y_bnd)
        v_pred = -self.gradients(psi, self.x_bnd)

        p_x = self.gradients(p, self.x_bnd)
        p_y = self.gradients(p, self.y_bnd)

        x, y = self.x_bnd, self.y_bnd
        inlet = (y == self.ub[1])
        h_wall = (y == self.lb[1])
        v_wall = (x == self.lb[0]) | (x == self.ub[0])
        p_loss = torch.hstack([p[inlet], p_x[v_wall], p_y[h_wall]]).reshape(-1, 1)

        mse_p = torch.mean(p_loss**2)
        mse_u = self.mse(u_pred, self.u_bnd)
        mse_v = self.mse(v_pred, self.v_bnd)
        return mse_p + mse_u + mse_v
    
    def ic_loss(self):
        res = self.forward(torch.hstack([self.x_ic, self.y_ic, self.t_ic]))
        psi, p = res[:, 0:1], res[:, 1:2]
        u_pred = self.gradients(psi, self.y_ic)
        v_pred = -self.gradients(psi, self.x_ic)

        mse_p = torch.mean(p**2)
        mse_u = self.mse(u_pred, self.u_ic)
        mse_v = self.mse(v_pred, self.v_ic)
        return mse_p + mse_u + mse_v

    def loss_function(self):
        mse_pde = self.pde_loss()
        mse_bnd = self.bc_loss()
        mse_ic = self.ic_loss()
        
        loss = self.w_f * mse_pde + self.w_bc * mse_bnd + self.w_ic * mse_ic
        return loss, mse_pde.item(), mse_bnd.item(), mse_ic.item()

    def train(self, X_f, X_bnd, U_bnd, X_ic, U_ic, lr=1e-3, epochs=1000, adam=True, lbfgs=True):
        self.x_f, self.y_f, self.t_f = self.set_inputs(X_f, requires_grad=True)
        self.x_bnd, self.y_bnd, self.t_bnd = self.set_inputs(X_bnd, requires_grad=True)
        self.x_ic, self.y_ic, self.t_ic = self.set_inputs(X_ic, requires_grad=True)

        self.u_bnd = self.to_tensor(U_bnd[:, 0:1])
        self.v_bnd = self.to_tensor(U_bnd[:, 1:2])

        self.u_ic = self.to_tensor(U_ic[:, 0:1])
        self.v_ic = self.to_tensor(U_ic[:, 1:2])

        if not adam and not lbfgs:
            raise ValueError("At least one optimizer (Adam or L-BFGS) must be selected.")

        self.configure_optimizers(lr=lr)
        self.start_time = perf_counter()

        if adam:
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
                    hours = elapsed // 3600
                    minutes = (elapsed % 3600) // 60
                    seconds = elapsed % 60

                    print(f"[Adam] Epoch {epoch}/{epochs}, "
                        f"Loss = {loss.item():.6e}, "
                        f"pde = {mse_pde:.3e}, "
                        f"ic = {mse_ic:.3e}, "
                        f"bc = {mse_bnd:.3e}, "
                        f"Time = {elapsed:.2f} s, "
                        f"hh:mm:ss = {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")

        if lbfgs:
            loss = self.optimizer_lbfgs.step(self.closure)
            elapsed = perf_counter() - self.start_time
            self.end_time = elapsed

            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60

            print(f"[LBFGS] Optimization finished, final values\n"
                f"[LBFGS] Iter {self.lbfgs_step}, "
                f"Loss = {self.loss_history['total'][-1]:.6e}, "
                f"pde = {self.loss_history['pde'][-1]:.3e}, "
                f"ic = {self.loss_history['ic'][-1]:.3e}, "
                f"bc = {self.loss_history['bc'][-1]:.3e}, "
                f"Time = {self.end_time:.2f} s"
                f"hh:mm:ss = {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")


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

            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60

            print(f"[LBFGS] Iter {self.lbfgs_step}, "
                    f"Loss = {loss.item():.6e}, "
                    f"pde = {mse_pde:.3e}, "
                    f"ic = {mse_ic:.3e}, "
                    f"bc = {mse_bnd:.3e}, "
                    f"Time = {elapsed:.2f} s"
                    f"hh:mm:ss = {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}")

        self.lbfgs_step += 1
        return loss

    def predict(self, x, y, t):
        x = self.to_tensor(x, requires_grad=True).reshape(-1, 1)
        y = self.to_tensor(y, requires_grad=True).reshape(-1, 1)
        t = self.to_tensor(t, requires_grad=True).reshape(-1, 1)

        res = self.forward(torch.hstack([x, y, t]))
        psi, p = res[:, 0:1], res[:, 1:2]
        
        u = self.gradients(psi, y)
        v = -self.gradients(psi, x)

        return (
            u.detach().cpu().numpy(),
            v.detach().cpu().numpy(),
            p.detach().cpu().numpy()
        )

