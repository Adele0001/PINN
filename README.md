# PINN — Continuous-Time Physics-Informed Neural Network for Burgers' Equation

Reproduces: Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks: A
deep learning framework for solving forward and inverse problems involving nonlinear
partial differential equations*, Appendix A.1 (Fig. A.6 / Table A.1 benchmark).

Implemented in PyTorch (the original paper's code was TensorFlow 1.x).

## What this project does

Trains a neural network to solve Burgers' equation:

```
u_t + u*u_x - (0.01/pi)*u_xx = 0,   x in [-1,1], t in [0,1]
u(0,x) = -sin(pi*x)
u(t,-1) = u(t,1) = 0
```

using only 100 known initial/boundary points plus 10,000 unlabeled "collocation"
points where the PDE itself is enforced — rather than being trained on the full
solution directly. Expected relative L2 error vs. the exact solution: ~6.7e-4
(the paper's reported benchmark for this configuration).

## Repository contents

```
research-internship/
├── PINNs/
│   └── appendix/
│       └── Data/
│           └── burgers_shock.mat      <- reference solution (required to run)
├── PINN-Implementation.ipynb      <- the notebook with implementations
├── requirements.txt
└── README.md
```

Note: `venv/`, `*.pt` checkpoints, and `__pycache__/` are excluded via
`.gitignore` and are *not* included in this repo — see setup steps below for
how to regenerate them.

## Setup: cloning and running this project from scratch

### 1. Clone the repository

```bash
git clone https://github.com/Adele0001/PINN.git
cd PINN
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows
```

You should see `(venv)` appear at the start of your terminal prompt once
activated.

### 3. Install dependencies

```bash
pip install torch jupyter numpy matplotlib scipy
```

(If a `requirements.txt` is present in the repo, you can instead run
`pip install -r requirements.txt` to install pinned versions.)

### 4. Confirm the reference data file is present

The notebook expects to find `burgers_shock.mat` at a specific path. Check
that it exists:

```bash
ls PINNs/appendix/Data/burgers_shock.mat
```

If it's missing, it needs to be obtained separately (e.g. from the original
paper's own GitHub repository, [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs),
under `main/Data/burgers_shock.mat`) and placed at that exact path.

### 5. Update the hardcoded file path (if needed)

The notebook currently loads the data using an **absolute path** specific to
the original author's machine:

```python
data = loadmat("/Users/adiljan/desktop/research-internship/PINNs/appendix/Data/burgers_shock.mat")
```

If you're running this on a different machine or from a different folder
location, update this line to point to wherever `burgers_shock.mat` actually
sits on your system — otherwise the notebook will fail at Section 1 with a
file-not-found error.

### 6. Launch Jupyter

```bash
jupyter notebook
```

or, if you prefer Jupyter Lab:

```bash
jupyter lab
```

This opens a browser window. From there, open `PINN-Implementation.ipynb`.

### 7. Run the notebook

From Jupyter's menu, select **Run → Run All Cells** (or, in the toolbar,
**Run All**), or step through cells one at a time with `Shift+Enter`.

Training (Section 7) takes roughly 1–3 minutes on Apple Silicon with MPS
acceleration, longer on CPU-only machines. Progress is printed every 100
L-BFGS iterations.

### 8. Check the output

After training completes, you should see:
- A printed relative L2 error (compare it against the paper's reported
  `6.7e-4` benchmark — a result in the same order of magnitude, e.g.
  `5e-4` to `1e-3`, is expected and correct; exact reproduction isn't
  guaranteed due to random seed and training variability, as the paper
  itself acknowledges)
- A saved figure, `burgers_pinn_full_figure.png`, reproducing the paper's
  Fig. A.6 (heatmap + three time-snapshot comparisons)
- A saved model checkpoint and prediction array (see next section)

## Checkpointing and resuming training

This notebook supports resuming training across multiple sessions, which is
useful if a long run gets interrupted. This is handled by two variables near
the top of **Section 3** and the end of **Section 8**:

```python
CKPT = "pinn_burgers_model1.pt"
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT))
    print(f"Resumed from checkpoint {CKPT}")
```

and, at the end of training (Section 8):

```python
torch.save(model.state_dict(), "pinn_burgers_model1.pt")
```

**How this works in practice:**

- The **first time** you run the notebook, no file named `pinn_burgers_model2.pt`
  exists yet, so training starts from scratch (freshly initialized weights).
- At the end of the run, the trained weights are saved to that filename.
- If you **run the notebook again** without renaming anything, it will detect
  the existing checkpoint, print `Resumed from checkpoint pinn_burgers_model2.pt`,
  and continue training from those saved weights rather than starting over.

**If you want a clean, from-scratch training run** (e.g. to get a result
directly comparable to a fresh paper reproduction, or to demo the full
training process from zero):

```bash
rm pinn_burgers_model1.pt
```

Delete (or rename) this file *before* running the notebook, so Section 3
finds no existing checkpoint and initializes new random weights instead.

**If you want to keep multiple separate trained models** (e.g. to compare
different hyperparameter choices without overwriting each other), change the
`CKPT` variable to a new filename *before* running, for example:

```python
CKPT = "pinn_burgers_model_8layers.pt"
```

Each distinct filename you use creates (and later resumes from) its own
independent checkpoint, so you can keep several trained variants side by
side without them interfering with one another. Just remember to use the
exact same filename across multiple runs if your intent is to continue
training the *same* model rather than start a new one.

## Known limitations / things to double-check

- The `.mat` file path in Section 1 is hardcoded to one specific environment
  folder structure — update it per step 5 above before running elsewhere.
- Training uses `device = "mps"` if available (Apple Silicon GPU
  acceleration), falling back to CPU otherwise. If you hit unexpected errors
  or NaNs during training on an Apple Silicon Mac, try forcing
  `device = torch.device("cpu")` to rule out an MPS-specific issue.
- Relative L2 error will vary slightly between runs due to random
  initialization and point sampling, even with a fixed seed, if run on
  different hardware/backends — this matches variability the original
  paper's authors report themselves.
