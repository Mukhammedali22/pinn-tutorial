import numpy as np
from numpy import (pi, exp, sin, cos)
import matplotlib.pyplot as plt
from time import perf_counter, sleep


def Jacobi_method(P:np.ndarray, U:np.ndarray, V:np.ndarray, 
                  N, M, dx, dy, dt, rho) -> np.ndarray:
    """Jacobi method for solving 2D Poisson equation"""
    # Reference
    P_old = P
    U_new = U
    V_new = V

    P_new = np.zeros_like(P)

    P_new[1:M-1, 1:N-1] = (dy**2*(P_old[1:M-1, 2:N] + P_old[1:M-1, 0:N-2]) \
        + dx**2*(P_old[2:M, 1:N-1] + P_old[0:M-2, 1:N-1])) \
            / (2*(dx**2 + dy**2)) \
        - dx**2*dy**2*rho \
            / (2*dt*(dx**2 + dy**2)) \
        * ((U_new[1:M-1, 2:N] - U_new[1:M-1, 0:N-2]) \
            / (2*dx) \
        + (V_new[2:M, 1:N-1] - V_new[0:M-2, 1:N-1]) \
            / (2*dy))
    
    return P_new


def Burger_explicit(S:np.ndarray, U:np.ndarray, V:np.ndarray, 
                    N, M, dx, dy, dt, nu) -> np.ndarray:
    # Reference
    U_old = U
    V_old = V
    S_old = S
    
    S_new = np.zeros_like(S)
    
    S_new[1:M-1, 1:N-1] = S_old[1:M-1, 1:N-1] + dt*(
        - U_old[1:M-1, 1:N-1]*(S_old[1:M-1, 2:N] - S_old[1:M-1, 0:N-2]) \
                                / (2*dx) \
        - V_old[1:M-1, 1:N-1]*(S_old[2:M, 1:N-1] - S_old[0:M-2, 1:N-1]) \
                                / (2*dy) \
        + nu*(
            (S_old[1:M-1, 2:N] - 2*S_old[1:M-1, 1:N-1] + S_old[1:M-1, 0:N-2]) \
                / dx**2 \
            + (S_old[2:M, 1:N-1] - 2*S_old[1:M-1, 1:N-1] + S_old[0:M-2, 1:N-1]) \
                / dy**2))
    
    return S_new

def set_boundary_U(U:np.ndarray):
    U[:, 0] = 0 # x=0 u=0
    U[:, -1] = 0 # x=1 u=0
    U[0, :] = 0 # y=0 u=0
    U[-1, :] = 1 # y=1 u=1
    
def set_boundary_V(V:np.ndarray):
    V[:, 0] = 0 # x=0 v=0
    V[:, -1] = 0 # x=1 v=0
    V[0, :] = 0 # y=0 v=0
    V[-1, :] = 0 # y=1 v=0
    
def set_boundary_P(P:np.ndarray):
    P[:, 0] = P[:, 1] # x=0 dp/dx=0
    P[:, -1] = P[:, -2] # x=1 dp/dx=0
    P[0, :] = P[1, :] # y=0 dp/dy=0
    P[-1, :] = 0 # y=1 p=0
    



if __name__ == "__main__":
    start_x, end_x = (0, 1)
    start_y, end_y = (0, 1)

    N = 100
    M = 100
    dx = (end_x - start_x) / (N - 1)
    dy = (end_y - start_y) / (M - 1)

    Time = 1 # 1 second
    dt = 0.001
    t_iter = 0.01 / dt
    stop_iteration = int(1 / dt) + 1
    Re = 100
    rho = 1
    eps = 1e-6
    eps_P = 1e-6
    stop_iteration_P = 1e5

    x = start_x + np.arange(start=0, stop=N) * dx
    y = start_y + np.arange(start=0, stop=M) * dy
    X, Y = np.meshgrid(x, y)


    U_old = np.zeros((M, N))
    U_new = np.zeros((M, N))

    V_old = np.zeros((M, N))
    V_new = np.zeros((M, N))

    P_old = np.zeros((M, N))
    P_new = np.zeros((M, N))

    set_boundary_U(U=U_old)
    set_boundary_V(V=V_old)
    set_boundary_P(P=P_old)

    file_name = "num_sol.csv"
    try:
        with open(file_name, 'w') as f:
            print("Okay")
    except:
        pass

    start_time = perf_counter()
    maximum = 1
    for i in range(int(stop_iteration)):

        U_new = Burger_explicit(U_old, U_old, V_old, N, M, dx, dy, dt, 1/Re)
        V_new = Burger_explicit(V_old, U_old, V_old, N, M, dx, dy, dt, 1/Re)
        set_boundary_U(U=U_new)
        set_boundary_V(V=V_new)

        set_boundary_P(P=P_old)
        maximum_P = 1
        for j in range(int(stop_iteration_P)):
            P_new = Jacobi_method(P_old, U_new, V_new, N, M, dx, dy, dt, rho)
            set_boundary_P(P=P_new)

            maximum_P = np.max(np.abs(P_new - P_old))
            P_old = P_new.copy()

            if maximum_P < eps_P:
                # print(f"Pressure field converged after {j} iteration")
                break


        # Velocity correction
        U_new[1:M-1, 1:N-1] = U_new[1:M-1, 1:N-1] - dt / rho*(
            P_new[1:M-1, 2:N] - P_new[1:M-1, 0:N-2]) \
                / (2*dx)
        
        V_new[1:M-1, 1:N-1] = V_new[1:M-1, 1:N-1] - dt / rho*(
            P_new[2:M, 1:N-1] - P_new[0:M-2, 1:N-1]) \
                / (2*dy)

        set_boundary_U(U=U_new)
        set_boundary_V(V=V_new)

        max_U = np.max(np.abs(U_new - U_old))
        max_V = np.max(np.abs(V_new - V_old))
        maximum = max(max_U, max_V)

        if i % 10 == 0:
            print(f"Max abs diff: {maximum:.6e} at iter: {i} Pressure converged at iter: {j}")

        U_old = U_new.copy()
        V_old = V_new.copy()

        # writing results
        if i % t_iter == 0:
            print(f"\n\n{'-'*20}Saving results at t = {i * dt:.2f} s{'-'*20}\n\n")
            
            header = "T,U,V,P"
            
            # [t,u,v,p] format
            data = np.hstack([i*dt*np.ones_like(X).reshape(-1, 1), 
                            U_new.reshape(-1, 1), V_new.reshape(-1, 1), P_new.reshape(-1, 1)]).copy()
            
            file_name = "num_sol.csv"
            try:
                with open(file_name, 'x') as f:
                    np.savetxt(f, data, delimiter=',', fmt='%.6e', header=header)
            except FileExistsError:        
                with open(file_name, 'a') as f:
                    np.savetxt(f, data, delimiter=',', fmt='%.6e')

            # sleep(0.5)

        if maximum < eps:
            print(f"Velocity field converged after {i} iteration")
            break

    end_time = perf_counter()
    print(f"Calculation time: {end_time - start_time:.6f} seconds")

    # post process

    def draw_all(results, names, arrow=True):
        fig, ax = plt.subplots(2, 2)
        cnt = 0
        for row in range(2):
            for col in range(2):
                axs = ax[row, col]
                cf = axs.contourf(X, Y, results[cnt % 4])
                fig.colorbar(cf, ax=axs)
                axs.set_title(names[cnt % 4])
                if arrow:
                    axs.streamplot(X, Y, U_new, V_new, color="black")
                axs.set_xlabel("x")
                axs.set_ylabel("y")
                cnt += 1
        plt.tight_layout()
        plt.show()


    def draw_one(Z, name, arrow=False):
        fig, ax = plt.subplots()
        cf = ax.contourf(X, Y, Z)
        fig.colorbar(cf, ax=ax)
        if arrow:
            ax.streamplot(X, Y, U_new, V_new, color="black")
        plt.title(name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


    UV = (U_new**2 + V_new**2)**0.5
    data = [U_new, V_new, P_new, UV]
    name = ["U", "V", "P", "U + V"]
    draw_all(data, name, arrow=True)