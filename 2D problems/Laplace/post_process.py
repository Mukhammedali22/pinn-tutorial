import torch 
import numpy as np
import matplotlib.pyplot as plt
from main import NeuralNet, LaplacePinn


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


x_bounds = (0.0, 1.0)
y_bounds = (0.0, 1.0)

lb = [x_bounds[0], y_bounds[0]]
ub = [x_bounds[1], y_bounds[1]]


model = NeuralNet(input_dim=2, output_dim=1, hidden_layers=6, hidden_dim=50)
pinn = LaplacePinn(model, w_f=1.0, w_bc=100.0)

pinn.load_state_dict(torch.load("trained_model.pt"))

nx, ny = (100, 100)
x_test = np.linspace(lb[0], ub[0], nx)
y_test = np.linspace(lb[1], ub[1], ny)
X, Y = np.meshgrid(x_test, y_test)

u_pred = pinn.predict(X, Y).reshape(X.shape)

u_num = Jacobi_method(nx, ny)
# error = np.abs(u_pred[1:-1, 1:-1] - u_num[1:-1, 1:-1])
error = np.abs(u_pred[1:-1, 1:-1] - u_num[1:-1, 1:-1])
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
# plt.savefig('Figure 1.png')
plt.show()
