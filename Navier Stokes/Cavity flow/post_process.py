import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from network import PINN


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_contour_field(X, Y, field, U=None, V=None, title="", cmap="turbo", save_path=None):
    """
    Plot a scalar field using contourf, optionally with streamlines.

    Parameters
    ----------
    X, Y : 2D meshgrid arrays
    field : 2D array
        Scalar field to visualize
    U, V : 2D arrays, optional
        Velocity components for streamlines
    title : str
        Plot title
    cmap : str
        Colormap
    save_path : str, optional
        If provided, saves the figure instead of showing
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.contourf(X, Y, field, levels=100, cmap=cmap)
    fig.colorbar(cax, ax=ax, label=title if title else "Field")

    if U is not None and V is not None:
        ax.streamplot(X, Y, U, V, color="k", density=1.5, linewidth=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    
    plt.show()


def make_field_animation(X, Y, field, t_values, title="Field Evolution",
                         cmap="turbo", save_path="field_animation.gif",
                         interval=200, U=None, V=None, stream_density=1.2):
    """
    Create an animation of a scalar field (precomputed) with optional streamlines.

    Parameters
    ----------
    X, Y : 2D meshgrid arrays
    field : 3D array (nt, ny, nx)
        Field values for each time step
    t_values : 1D array
        Corresponding time values
    title : str
        Animation title
    cmap : str
        Colormap
    save_path : str
        Output GIF file name
    interval : int
        Frame duration in milliseconds
    U, V : 3D arrays (nt, ny, nx), optional
        Velocity components for streamlines. If None, no streamlines are plotted.
    stream_density : float
        Density of streamlines (passed to streamplot)
    """
    nt = len(t_values)
    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.contourf(X, Y, field[0], levels=100, cmap=cmap)
    cb = fig.colorbar(cax, ax=ax, label=title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} at t={t_values[0]:.2f}")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    def update(frame):
        ax.clear()
        cax = ax.contourf(X, Y, field[frame], levels=100, cmap=cmap)
        if U is not None and V is not None:
            ax.streamplot(X, Y, U[frame], V[frame],
                          color="k", linewidth=0.8, density=stream_density)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title} at t={t_values[frame]:.2f}")
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        cb.ax.clear()
        fig.colorbar(cax, ax=ax, cax=cb.ax, label=title)

    ani = animation.FuncAnimation(fig, update, frames=nt,
                                  interval=interval, blit=False)
    ani.save(filename=save_path, writer="pillow", fps=10)
    plt.close(fig)


def plot_all_fields(X, Y, u, v, p, title_prefix="", save_path=None):
    """
    Plot u, v, and p fields side by side (like plot_contour_field but for 3 fields).

    Parameters:
        X, Y : 2D meshgrid arrays
        u, v, p : 2D field arrays
        title_prefix : optional string prefix for titles (e.g., 't = 0.5s')
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fields = [u, v, p]
    titles = [f"{title_prefix} u", f"{title_prefix} v", f"{title_prefix} p"]

    for ax, field, title in zip(axs, fields, titles):
        cf = ax.contourf(X, Y, field, levels=100, cmap='jet')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        fig.colorbar(cf, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    
    plt.show()


def l2_relative_error(pred, true):
    errors = []
    for i in range(pred.shape[0]):
        num = np.linalg.norm(pred[i].ravel() - true[i].ravel(), 2)
        denom = np.linalg.norm(true[i].ravel(), 2)
        errors.append(num / denom)
    return np.array(errors)


x_bounds = (0.0, 1.0)
y_bounds = (0.0, 1.0)
t_bounds = (0.0, 1.0)

lb = np.array([x_bounds[0], y_bounds[0], t_bounds[0]])
ub = np.array([x_bounds[1], y_bounds[1], t_bounds[1]])

layers = [3, 50, 50, 50, 50, 2]
pinn = PINN(lb, ub, layers=layers)
pinn.load_model("NS_pinn_model.pt")


nx, ny = (100, 100)
x_test = np.linspace(lb[0], ub[0], nx)
y_test = np.linspace(lb[1], ub[1], ny)
X, Y = np.meshgrid(x_test, y_test)

data = np.loadtxt("num_sol.csv", delimiter=",")
t_values = np.unique(data[:, 0])
nt = len(t_values)

t_num = data[:, 0].reshape(nt, ny, nx)
u_num = data[:, 1].reshape(nt, ny, nx)
v_num = data[:, 2].reshape(nt, ny, nx)
p_num = data[:, 3].reshape(nt, ny, nx)

u_pred = np.zeros_like(u_num)
v_pred = np.zeros_like(v_num)
p_pred = np.zeros_like(p_num)

for i in range(nt):
    u, v, p = pinn.predict(x=X, y=Y, t=t_num[i])
    u_pred[i] = u.reshape(X.shape)
    v_pred[i] = v.reshape(X.shape)
    p_pred[i] = p.reshape(X.shape)

l2_u = l2_relative_error(u_pred, u_num)
l2_v = l2_relative_error(v_pred, v_num)
l2_p = l2_relative_error(p_pred, p_num)

mean_u, mean_v, mean_p = np.mean(l2_u), np.mean(l2_v), np.mean(l2_p)
print(f"Mean relative L2 error:")
print(f"U: {mean_u:.4e}, V: {mean_v:.4e}, P: {mean_p:.4e}")

total_l2_u = np.linalg.norm(u_pred - u_num) / np.linalg.norm(u_num)
total_l2_v = np.linalg.norm(v_pred - v_num) / np.linalg.norm(v_num)
total_l2_p = np.linalg.norm(p_pred - p_num) / np.linalg.norm(p_num)
print(f"Global relative L2 error: U={total_l2_u:.4e}, V={total_l2_v:.4e}, P={total_l2_p:.4e}")

plt.figure(figsize=(7, 5))
plt.plot(t_values, l2_u, label=r'$L_2(u)$', lw=2)
plt.plot(t_values, l2_v, label=r'$L_2(v)$', lw=2)
plt.plot(t_values, l2_p, label=r'$L_2(p)$', lw=2)
plt.xlabel("Time")
plt.ylabel(r"Relative $L_2$ Error")
plt.title("L2 Error Evolution over Time")
plt.legend()
plt.grid(True, ls="--", alpha=0.6)
plt.tight_layout()
plt.show()

np.savetxt("Output\\U_num.txt", u_num[-1], delimiter='\t', fmt='%.6f')
np.savetxt("Output\\V_num.txt", v_num[-1], delimiter='\t', fmt='%.6f')
np.savetxt("Output\\P_num.txt", p_num[-1], delimiter='\t', fmt='%.6f')

np.savetxt("Output\\U_pinn.txt", u_pred[-1], delimiter='\t', fmt='%.6f')
np.savetxt("Output\\V_pinn.txt", v_pred[-1], delimiter='\t', fmt='%.6f')
np.savetxt("Output\\P_pinn.txt", p_pred[-1], delimiter='\t', fmt='%.6f')

np.savetxt("Output\\X.txt", X, delimiter='\t', fmt='%.4f')
np.savetxt("Output\\Y.txt", Y, delimiter='\t', fmt='%.4f')

plot_contour_field(X, Y, u_pred[-1], title="PINN U at t=1.0", save_path="Output\\u_pinn.png")
plot_contour_field(X, Y, v_pred[-1], title="PINN V at t=1.0", save_path="Output\\v_pinn.png")
plot_contour_field(X, Y, p_pred[-1], title="PINN P at t=1.0", save_path="Output\\p_pinn.png")

plot_contour_field(X, Y, np.abs(u_pred[-1] - u_num[-1]), title="Absolute error at t=1.0",
                    save_path="Output\\Absolute error plot.png")

plot_all_fields(X, Y, u_pred[-1], v_pred[-1], p_pred[-1], title_prefix="t = 1s", 
                    save_path="Output\\uvp.png")

make_field_animation(X, Y, u_pred, t_values, U=u_pred, V=v_pred, 
                     title="PINN U(t,x,y)", save_path="Output\\U_pred.gif")
make_field_animation(X, Y, v_pred, t_values, title="PINN V(t,x,y)", save_path="Output\\V_pred.gif")
make_field_animation(X, Y, p_pred, t_values, title="PINN P(x,y)", save_path="Output\\P_pred.gif")