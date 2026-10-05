# Sparse Linear MPC QP Generator

A Python implementation for formulating and generating sparse Quadratic Programming (QP) matrices for discrete-time Linear Model Predictive Control (LMPC). Built with `pydantic` for strict validation and `scipy.sparse` for scalable, memory-efficient matrix assembly.

## Discrete LMPC Formulation

Consider a discrete-time linear time-invariant system:
$$ x_{k+1} = A x_k + B u_k $$

Given a prediction horizon $N$, state dimension $n_x$, and input dimension $n_u$, the LMPC solves the following optimal control problem at each time step:

$$ 
\begin{aligned}
\min_{x, u} \quad & \sum_{k=1}^{N} \left( (x_k - x_{\text{ref},k})^T Q (x_k - x_{\text{ref},k}) + (u_k - u_{\text{ref},k})^T R (u_k - u_{\text{ref},k}) \right) \\
& + (x_{N+1} - x_{\text{ref},N+1})^T Q_{\text{end}} (x_{N+1} - x_{\text{ref},N+1}) \\
\text{s.t.} \quad & x_1 = \bar{x}_0 \\
& x_{k+1} = A x_k + B u_k, \quad \forall k \in \{1, \dots, N\} \\
& x_{lb} \le x_k \le x_{ub}, \quad \forall k \in \{1, \dots, N+1\} \\
& u_{lb} \le u_k \le u_{ub}, \quad \forall k \in \{1, \dots, N\}
\end{aligned}
$$

where $\bar{x}_0$ is the measured initial state.

## Sparse QP Formulation

The problem is transcribed into a standard sparse QP:
$$ 
\begin{aligned}
\min_{w} \quad & \frac{1}{2} w^T H w + c^T w \\
\text{s.t.} \quad & A_{qp} w = b_{qp} \\
& lb \le w \le ub
\end{aligned}
$$

The decision variable is stacked as $w = [x_1^T, \dots, x_{N+1}^T, u_1^T, \dots, u_N^T]^T \in \mathbb{R}^{(N+1)n_x + N n_u}$. 

The QP matrices are constructed as follows:

### Hessian ($H$)
Block-diagonal matrix containing the stage and terminal costs:
$$ H = \text{blkdiag}(\underbrace{Q, \dots, Q}_{N}, Q_{\text{end}}, \underbrace{R, \dots, R}_{N}) $$

### Linear Cost Vector ($c$)
Derived from the reference trajectories $w_{\text{ref}} = [x_{\text{ref},1}^T, \dots, x_{\text{ref},N+1}^T, u_{\text{ref},1}^T, \dots, u_{\text{ref},N}^T]^T$:
$$ c = -H w_{\text{ref}} $$

### Equality Constraints ($A_{qp}, b_{qp}$)
The dynamics and initial condition are enforced via $A_{qp} = \begin{bmatrix} C_x & C_u \end{bmatrix}$:

$$ 
C_x = I_{(N+1)n_x} - \mathcal{L} \otimes A 
$$
where $\mathcal{L} \in \mathbb{R}^{(N+1) \times (N+1)}$ is the lower-shift matrix (1s on the first sub-diagonal, 0s elsewhere), and $\otimes$ denotes the Kronecker product.

$$ 
C_u = \begin{bmatrix} 0_{n_x \times N n_u} \\ -I_N \otimes B \end{bmatrix} 
$$

The right-hand side enforces the initial state $\bar{x}_0$:
$$ 
b_{qp} = \begin{bmatrix} \bar{x}_0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} 
$$

### Box Constraints ($lb, ub$)
Bounds are repeated across the horizon:
$$ 
lb = \begin{bmatrix} \mathbf{1}_{N+1} \otimes x_{lb} \\ \mathbf{1}_N \otimes u_{lb} \end{bmatrix}, \quad 
ub = \begin{bmatrix} \mathbf{1}_{N+1} \otimes x_{ub} \\ \mathbf{1}_N \otimes u_{ub} \end{bmatrix} 
$$

## Usage

```python
import numpy as np
from linear_mpc import LinearMPC

# 1. Define system dimensions and horizon
nx, nu, N = 2, 1, 10

# 2. Initialize matrices (numpy arrays)
A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])
Q = np.eye(nx)
R = 0.1 * np.eye(nu)
Qend = np.eye(nx)

# 3. Define references and bounds
x0 = np.array([[1.0], [0.0]])
x_ref = np.zeros((nx, N + 1))
u_ref = np.zeros((nu, N))
x_lb, x_ub = -10 * np.ones((nx, 1)), 10 * np.ones((nx, 1))
u_lb, u_ub = -1 * np.ones((nu, 1)), 1 * np.ones((nu, 1))

# 4. Instantiate and generate sparse QP
mpc = LinearMPC(
    intervals=N, nx=nx, nu=nu,
    A=A, B=B, Q=Q, R=R, Qend=Qend,
    x0=x0, x_ref=x_ref, u_ref=u_ref,
    x_lb=x_lb, x_ub=x_ub, u_lb=u_lb, u_ub=u_ub
)

qp_data = mpc.generate_QP()
# qp_data contains scipy.sparse.csc_matrix for 'H', 'A' and numpy arrays for 'c', 'b', 'lb', 'ub'
```

## Dependencies
- `numpy`
- `scipy`
- `pydantic`
