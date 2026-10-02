"""
Continuous-time Physics-Informed Neural Network (PINN) for Burgers' equation.

Reproduces: Raissi, Perdikaris & Karniadakis (2019), Appendix A.1,
Fig. A.6 / Table A.1 benchmark:
    N_u = 100 initial/boundary points, N_f = 10,000 collocation points,
    9 hidden layers x 20 neurons, tanh activations, trained with L-BFGS.
    Expected relative L2 error vs. exact solution: ~6.7e-4

PDE (Burgers' equation):
    u_t + u*u_x - (0.01/pi)*u_xx = 0,   x in [-1,1], t in [0,1]
    u(0,x) = -sin(pi*x)
    u(t,-1) = u(t,1) = 0
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc
from scipy.io import loadmat

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# 1. Load reference data
# -----------------------------------------------------------------------
data = loadmat("Data/burgers_shock.mat")
x = data["x"].flatten()[:, None]      # (256, 1)   spatial grid, in [-1, 1]
t = data["t"].flatten()[:, None]      # (100, 1)   time grid,   in [0, 1]
usol_exact = data["usol"]             # (256, 100) u(x_i, t_j)  -- rows=x, cols=t

nu_pde = 0.01 / np.pi                 # diffusion coefficient in the PDE

# -----------------------------------------------------------------------
# 2. Build training data: IC/BC points (N_u) + collocation points (N_f)
# -----------------------------------------------------------------------
N_u = 100
N_f = 10000

X, T = np.meshgrid(x, t)              # X, T shape: (100, 256) i.e. (len(t), len(x))
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))   # all (x,t) pairs
u_star = usol_exact.T.flatten()[:, None]   # transpose so it matches X_star ordering

lb = X_star.min(0)   # domain lower bound  [x_min, t_min]
ub = X_star.max(0)   # domain upper bound  [x_max, t_max]

# --- Initial condition: t = 0, all x, u = -sin(pi*x) ---
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))     # (256, 2): (x, t=0)
uu1 = usol_exact[:, 0:1]                        # (256, 1): u(x, 0)

# --- Boundary condition: x = -1, all t, u = 0 ---
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))         # (100, 2): (x=-1, t)
uu2 = usol_exact[0:1, :].T                      # (100, 1)

# --- Boundary condition: x = 1, all t, u = 0 ---
xx3 = np.hstack((X[:, -1:], T[:, -1:]))         # (100, 2): (x=1, t)
uu3 = usol_exact[-1:, :].T                      # (100, 1)

X_u_train = np.vstack([xx1, xx2, xx3])          # all IC/BC candidate points
u_train_all = np.vstack([uu1, uu2, uu3])

# Randomly sample N_u of these IC/BC points (paper: "randomly distributed" data)
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train_all[idx, :]

# Collocation points via Latin Hypercube Sampling over the full domain
sampler = qmc.LatinHypercube(d=2, seed=1234)
sample = sampler.random(n=N_f)
X_f_train = lb + (ub - lb) * sample             # scale unit cube to [lb, ub]
X_f_train = np.vstack([X_f_train, X_u_train])   # standard practice: include IC/BC pts too

# -----------------------------------------------------------------------
# 3. Neural network: 2 inputs (x,t) -> 9 hidden layers x 20 neurons -> 1 output (u)
# -----------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)

        modules = []
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(linear.weight)   # Xavier init, as in paper's TF1 code
            nn.init.zeros_(linear.bias)
            modules.append(linear)
            modules.append(nn.Tanh())
        out_linear = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(out_linear.weight)
        nn.init.zeros_(out_linear.bias)
        modules.append(out_linear)

        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        # normalize inputs to [-1, 1] -- standard PINN practice for tanh net stability
        X = torch.cat([x, t], dim=1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(X)


layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 20, 1]   # 9 hidden layers x 20 neurons
model = PINN(layers, lb, ub).to(device)

# Optional: resume from a checkpoint if you're training across multiple
# sessions (e.g. if a long background job gets interrupted). Just rerun
# this script -- it will pick up from where it left off.
import os
CKPT = "pinn_burgers_model.pt"
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT))
    print(f"Resumed from checkpoint {CKPT}")

# -----------------------------------------------------------------------
# 4. Physics-informed residual f(t,x) via automatic differentiation
# -----------------------------------------------------------------------
def net_u(x, t):
    return model(x, t)

def net_f(x, t):
    """ f := u_t + u*u_x - nu*u_xx    (should be ~0 everywhere if PDE is satisfied) """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = net_u(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    f = u_t + u * u_x - nu_pde * u_xx
    return f

# -----------------------------------------------------------------------
# 5. Convert data to torch tensors
# -----------------------------------------------------------------------
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=device)

x_u = to_tensor(X_u_train[:, 0:1])
t_u = to_tensor(X_u_train[:, 1:2])
u_target = to_tensor(u_train)

x_f = to_tensor(X_f_train[:, 0:1])
t_f = to_tensor(X_f_train[:, 1:2])

# -----------------------------------------------------------------------
# 6. Loss function: MSE_u (data) + MSE_f (physics residual)
# -----------------------------------------------------------------------
mse = nn.MSELoss()

def loss_fn():
    u_pred = net_u(x_u, t_u)
    f_pred = net_f(x_f, t_f)
    loss_u = mse(u_pred, u_target)
    loss_f = mse(f_pred, torch.zeros_like(f_pred))
    return loss_u + loss_f, loss_u, loss_f

# -----------------------------------------------------------------------
# 7. Train with L-BFGS (full-batch, quasi-Newton -- as in the paper)
# -----------------------------------------------------------------------
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=5000,
    max_eval=5000,
    history_size=50,
    tolerance_grad=1e-9,
    tolerance_change=1e-12,
    line_search_fn="strong_wolfe",
)

iteration = [0]

def closure():
    optimizer.zero_grad()
    loss, loss_u, loss_f = loss_fn()
    loss.backward()
    iteration[0] += 1
    if iteration[0] % 100 == 0:
        print(f"iter {iteration[0]:5d}  loss={loss.item():.4e}  "
              f"loss_u={loss_u.item():.4e}  loss_f={loss_f.item():.4e}")
    return loss

print("Training with L-BFGS ...")
optimizer.step(closure)

# -----------------------------------------------------------------------
# 8. Evaluate: relative L2 error against the FULL reference grid
# -----------------------------------------------------------------------
model.eval()
x_star = to_tensor(X_star[:, 0:1])
t_star = to_tensor(X_star[:, 1:2])
with torch.no_grad():
    u_pred_star = net_u(x_star, t_star).cpu().numpy()

error_u = np.linalg.norm(u_star - u_pred_star, 2) / np.linalg.norm(u_star, 2)
print(f"\nRelative L2 error: {error_u:.6e}")
print(f"Paper's reported benchmark (N_u=100, N_f=10000, 9x20): 6.7e-4")

np.save("u_pred_star.npy", u_pred_star)
torch.save(model.state_dict(), "pinn_burgers_model.pt")
