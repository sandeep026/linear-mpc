# Linear MPC Parser

This project provides a robust, Pydantic-validated parser that translates a finite-horizon linear Model Predictive Control (MPC) problem into a standard structured Quadratic Program (QP). It handles discrete-time zero-order hold (ZOH) models, reference tracking, and box constraints for both states and inputs.

## Quick Start

To implement the parser and generate your QP matrices, simply define your system parameters as NumPy arrays and initialize the `LinearMPC` model:

```python
import numpy as np
from your_module import LinearMPC

# 1. Define dimensions and horizon
nx, nu, N = 2, 1, 10

# 2. Initialize the parser with your system matrices and bounds
mpc_problem = LinearMPC(
    intervals=N,                 # Horizon length N
    nx=nx,
    nu=nu,
    A=np.eye(nx),                # State transition matrix
    B=np.zeros((nx, nu)),        # Input matrix
    Q=np.eye(nx),                # State cost
    R=np.eye(nu),                # Input cost
    Qend=np.eye(nx),             # Terminal state cost
    x0=np.zeros((nx, 1)),        # Initial state
    x_ref=np.zeros((nx, N + 1)), # State reference trajectory
    u_ref=np.zeros((nu, N)),     # Input reference trajectory
    x_lb=-np.ones((nx, 1)) * 10, # State lower bound
    x_ub=np.ones((nx, 1)) * 10,  # State upper bound
    u_lb=-np.ones((nu, 1)) * 5,  # Input lower bound
    u_ub=np.ones((nu, 1)) * 5    # Input upper bound
)

# 3. Generate the QP dictionary (H, c, A, b, lb, ub)
qp_matrices = mpc_problem.generate_QP()
```

## Discrete-Time Model and ZOH

The matrices $A$ and $B$ are the discrete-time prediction model.

For a continuous-time model

$$
\dot{x}(t) = A_c x(t) + B_c u(t)
$$

a discrete model can be obtained with a zero-order hold (ZOH). Under ZOH, the control input is held constant during each sampling interval, giving

$$
x_{k+1} = A x_k + B u_k.
$$

This class uses the discrete matrices $A$ and $B$.

## LMPC Problem

The finite-horizon LMPC problem uses the following dimensions.

| Component            | Symbol      | Code Parameter | Dimension          |
| -------------------- | ----------- | -------------- | ------------------ |
| Horizon length       | $N$         | `intervals`    | scalar             |
| State dimension      | $n_x$       | `nx`           | scalar             |
| Control dimension    | $n_u$       | `nu`           | scalar             |
| State matrix         | $A$         | `A`            | $n_x \times n_x$   |
| Input matrix         | $B$         | `B`            | $n_x \times n_u$   |
| State cost matrix    | $Q$         | `Q`            | $n_x \times n_x$   |
| Input cost matrix    | $R$         | `R`            | $n_u \times n_u$   |
| Terminal cost matrix | $Q_{end}$   | `Qend`         | $n_x \times n_x$   |
| Initial state        | $\bar{x}_0$ | `x0`           | $n_x \times 1$     |
| State reference      | $x_{ref}$   | `x_ref`        | $n_x \times (N+1)$ |
| Input reference      | $u_{ref}$   | `u_ref`        | $n_u \times N$     |
| State lower bound    | $x_{lb}$    | `x_lb`         | $n_x \times 1$     |
| State upper bound    | $x_{ub}$    | `x_ub`         | $n_x \times 1$     |
| Input lower bound    | $u_{lb}$    | `u_lb`         | $n_u \times 1$     |
| Input upper bound    | $u_{ub}$    | `u_ub`         | $n_u \times 1$     |

The optimization variables are

$$
x_1,\ldots,x_{N+1},u_1,\ldots,u_N.
$$

The objective is

$$
\min \sum_{k=1}^{N} \Vert{}x_k-x_{ref,k}\Vert{}_Q^2
+ \Vert{}x_{N+1}-x_{ref,N+1}\Vert{}_{Q_{end}}^2
+ \sum_{k=1}^{N} \Vert{}u_k-u_{ref,k}\Vert{}_R^2
$$

with weighted norm

$$
\Vert{}z\Vert{}_Q^2 = z^T Q z.
$$

The terminal state uses $Q_{end}$ instead of $Q$.

The dynamics are

$$
x_1 = \bar{x}_0
$$

and

$$
x_{k+1} = A x_k + B u_k, \qquad k=1,\ldots,N.
$$

The state bounds are

$$
x_{lb} \leq x_k \leq x_{ub}, \qquad k=1,\ldots,N+1
$$

and the input bounds are

$$
u_{lb} \leq u_k \leq u_{ub}, \qquad k=1,\ldots,N.
$$

The cost matrices satisfy:

* $Q$ is symmetric positive semidefinite.
* $Q_{end}$ is symmetric positive semidefinite.
* $R$ is symmetric positive definite.

## Conversion to a QP

The LMPC problem is converted to a QP by stacking all state and control variables into one vector:

$$
w =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_{N+1} \\
u_1 \\
u_2 \\
\vdots \\
u_N
\end{bmatrix}.
$$

The dimension of $w$ is

$$
w \in \mathbb{R}^{(N+1)n_x + N n_u}.
$$

The conversion is done in three parts:

1. Construct the quadratic cost matrix $H_{QP}$ and linear cost vector $c_{QP}$.
2. Construct the equality constraint matrix $A_{QP}$ and vector $b_{QP}$.
3. Construct the lower and upper bounds $lb_{QP}$ and $ub_{QP}$.

A constant term that depends only on the references is ignored because it does not depend on $w$ and therefore does not change the optimizer.

## Cost Construction

For a state stage,

$$
\|x_k-x_{ref,k}\|_Q^2
=
x_k^T Q x_k
-
2 x_{ref,k}^T Q x_k
+
x_{ref,k}^T Q x_{ref,k}.
$$

For the terminal state,

$$
\|x_{N+1}-x_{ref,N+1}\|_{Q_{end}}^2
=
x_{N+1}^T Q_{end} x_{N+1}
-
2 x_{ref,N+1}^T Q_{end} x_{N+1}
+
\text{constant}.
$$

For a control stage,

$$
\|u_k-u_{ref,k}\|_R^2
=
u_k^T R u_k
-
2 u_{ref,k}^T R u_k
+
u_{ref,k}^T R u_{ref,k}.
$$

After stacking all variables, the quadratic cost matrix is

$$
H_{QP} = \operatorname{diag} \left( Q,\ldots,Q,Q_{end},R,\ldots,R \right).
$$

There are $N$ copies of $Q$, one copy of $Q_{end}$, and $N$ copies of $R$.

The linear cost vector is

$$
c_{QP}
=
\begin{bmatrix}
-2Q x_{ref,1} \\
-2Q x_{ref,2} \\
\vdots \\
-2Q x_{ref,N} \\
-2Q_{end} x_{ref,N+1} \\
-2R u_{ref,1} \\
-2R u_{ref,2} \\
\vdots \\
-2R u_{ref,N}
\end{bmatrix}.
$$

The resulting QP cost is

$$
\frac{1}{2} w^T H_{QP} w + c_{QP}^T w
$$

up to the constant reference-only term.

The factor of $1/2$ does not change the optimizer because the complete original LMPC objective has been scaled by the same positive constant during the QP construction.

## Equality Constraint Construction

The dynamics are written one step at a time.

The first equation is the initial condition:

$$
x_1 = \bar{x}_0.
$$

For each $k=1,\ldots,N$,

$$
x_{k+1} - A x_k - B u_k = 0.
$$

The state part of the equality matrix is


$$
C_x =
I_{N+1} \otimes I_{n_x}
-
L_{N+1} \otimes A,
$$

where $L_{N+1}$ is the strictly lower shift matrix

$$
L_{N+1} =
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
1 & 0 & \cdots & 0 \\
0 & 1 & \ddots & \vdots \\
\vdots & \ddots & \ddots & 0 \\
0 & \cdots & 1 & 0
\end{bmatrix}
$$


$$
C_u =
\begin{bmatrix}
0_{n_x \times N n_u} \\
-\left(I_N \otimes B\right)
\end{bmatrix}
$$

The complete equality matrix is

$$
A_{QP} =
\begin{bmatrix}
C_x & C_u
\end{bmatrix}.
$$

The right-hand side is

$$
b_{QP} =
\begin{bmatrix}
\bar{x}_0 \\
0 \\
\vdots \\
0
\end{bmatrix}.
$$

The first block row enforces $x_1=\bar{x}_0$. The remaining block rows enforce the dynamics for $k=1,\ldots,N$.
## Bound Construction

The state and input bounds are compactly represented using Kronecker products:

$$
lb_x = \mathbf{1}_{N+1} \otimes x_{lb},
\qquad
ub_x = \mathbf{1}_{N+1} \otimes x_{ub},
$$

$$
lb_u = \mathbf{1}_{N} \otimes u_{lb},
\qquad
ub_u = \mathbf{1}_{N} \otimes u_{ub}.
$$

The complete lower and upper bounds are

$$
lb_{QP} =
\begin{bmatrix}
lb_x \\
lb_u
\end{bmatrix},
\qquad
ub_{QP} =
\begin{bmatrix}
ub_x \\
ub_u
\end{bmatrix}.
$$
## QP Matrix and Vector Dimensions

Let

$$
n_w = (N+1)n_x + N n_u.
$$

Then the QP components have the following dimensions.

| Component             | Symbol    | Dimension             |
| --------------------- | --------- | --------------------- |
| Decision vector       | $w$       | $n_w \times 1$        |
| Quadratic cost matrix | $H_{QP}$  | $n_w \times n_w$      |
| Linear cost vector    | $c_{QP}$  | $n_w \times 1$        |
| Equality matrix       | $A_{QP}$  | $(N+1)n_x \times n_w$ |
| Equality RHS          | $b_{QP}$  | $(N+1)n_x \times 1$   |
| Lower bound           | $lb_{QP}$ | $n_w \times 1$        |
| Upper bound           | $ub_{QP}$ | $n_w \times 1$        |

The QP is therefore

$$
\min_w \frac{1}{2} w^T H_{QP} w + c_{QP}^T w
$$

subject to

$$
A_{QP} w = b_{QP}
$$

and

$$
lb_{QP} \leq w \leq ub_{QP}.
$$

