# Linear MPC

A lightweight implementation of **discrete-time linear Model Predictive Control (MPC)** that constructs the resulting optimization problem directly as a **sparse quadratic program (QP)** using NumPy and SciPy.

The implementation is designed for long prediction horizons where constructing dense lifted MPC matrices would be unnecessarily expensive in both memory and computation.

---

## Overview

Consider the discrete-time linear system

$$
x_{k+1}=Ax_k+Bu_k,
$$

with

$$
x_k\in\mathbb{R}^{n_x},
\qquad
u_k\in\mathbb{R}^{n_u}.
$$

Given a prediction horizon of \(N\) intervals, the MPC controller minimizes a quadratic tracking cost subject to the system dynamics and box constraints on states and inputs.

The implementation converts this MPC problem into the standard quadratic-programming form

$$
\begin{aligned}
\min_w\quad&
\frac12 w^T P w+q^T w\\
\text{s.t.}\quad&
A_{\mathrm{qp}}w=b_{\mathrm{qp}},\\
&
w_{\mathrm{lb}}\le w\le w_{\mathrm{ub}}.
\end{aligned}
$$

The resulting `P` and `A` matrices are sparse SciPy CSC matrices, while the vectors are dense NumPy arrays.

---

# 1. MPC problem

The controller solves

$$
\boxed{
\begin{aligned}
\min_{\{x_k,u_k\}}\quad&
\frac12
\sum_{k=0}^{N-1}
(x_k-x_k^{\mathrm{ref}})^T
Q
(x_k-x_k^{\mathrm{ref}})
\\
&+
\frac12
(x_N-x_N^{\mathrm{ref}})^T
Q_{\mathrm{end}}
(x_N-x_N^{\mathrm{ref}})
\\
&+
\frac12
\sum_{k=0}^{N-1}
(u_k-u_k^{\mathrm{ref}})^T
R
(u_k-u_k^{\mathrm{ref}})
\\[1mm]
\text{s.t.}\quad&
x_{k+1}=Ax_k+Bu_k,
\qquad k=0,\ldots,N-1,
\\
&
x_{\mathrm{lb}}\le x_k\le x_{\mathrm{ub}},
\qquad k=0,\ldots,N,
\\
&
u_{\mathrm{lb}}\le u_k\le u_{\mathrm{ub}},
\qquad k=0,\ldots,N-1,
\\
&
x_0=\hat{x}_0.
\end{aligned}
}
$$

Here:

* \(x_0=\hat{x}_0\) is the measured/current state.
* \(x_k^{\mathrm{ref}}\) is the desired state trajectory.
* \(u_k^{\mathrm{ref}}\) is the desired input trajectory.
* \(Q\succeq0\) weights state-tracking error.
* \(Q_{\mathrm{end}}\succeq0\) is the terminal-state weight.
* \(R\succ0\) weights input-tracking error.

There is no separate terminal set in this formulation. The terminal state is subject to the same state box bounds as the other predicted states.

---

# 2. Decision-vector construction

The implementation uses a **lifted formulation**, meaning that all predicted states and inputs are optimization variables.

The decision vector is

$$
\boxed{
w=
\begin{bmatrix}
x_0\\
x_1\\
\vdots\\
x_N\\
u_0\\
u_1\\
\vdots\\
u_{N-1}
\end{bmatrix}.
}
$$

Therefore the total number of decision variables is

$$
\boxed{
n_w=(N+1)n_x+Nn_u.
}
$$

This ordering is important because it determines the structure of every QP matrix.

For example, with

```text
N  = 1000
nx = 12
nu = 4
```

the decision vector contains

$$
(1000+1)12+1000(4)=16012
$$

variables.

---

# 3. From MPC cost to QP cost

Define the reference vector

$$
w_{\mathrm{ref}}=
\begin{bmatrix}
x_0^{\mathrm{ref}}\\
x_1^{\mathrm{ref}}\\
\vdots\\
x_N^{\mathrm{ref}}\\
u_0^{\mathrm{ref}}\\
\vdots\\
u_{N-1}^{\mathrm{ref}}
\end{bmatrix}.
$$

Define the block-diagonal weighting matrix

$$
H=
\operatorname{blkdiag}
\left(
\underbrace{Q,\ldots,Q}_{N},
Q_{\mathrm{end}},
\underbrace{R,\ldots,R}_{N}
\right).
$$

The tracking cost can then be written compactly as

$$
J=
\frac12
(w-w_{\mathrm{ref}})^T
H
(w-w_{\mathrm{ref}}).
$$

Expanding,

$$
J=
\frac12w^THw
-w_{\mathrm{ref}}^THw
+
\frac12w_{\mathrm{ref}}^THw_{\mathrm{ref}}.
$$

The final term depends only on the reference and therefore does not affect the optimizer.

Thus the QP can use

$$
\boxed{
P=H
}
$$

and

$$
\boxed{
q=-Hw_{\mathrm{ref}}.
}
$$

This gives the standard QP objective

$$
\boxed{
\frac12w^TPw+q^Tw.
}
$$

That is exactly the convention used by `qpsolvers`.

---

## Efficient construction of the linear term

There is no need to explicitly construct the large vector \(w_{\mathrm{ref}}\) and multiply the large sparse Hessian by it.

Because \(H\) is block diagonal,

$$
Hw_{\mathrm{ref}}
=
\begin{bmatrix}
Qx_0^{\mathrm{ref}}\\
\vdots\\
Qx_{N-1}^{\mathrm{ref}}\\
Q_{\mathrm{end}}x_N^{\mathrm{ref}}\\
Ru_0^{\mathrm{ref}}\\
\vdots\\
Ru_{N-1}^{\mathrm{ref}}
\end{bmatrix}.
$$

The implementation therefore computes

```python
cqp = -np.concatenate(
    (
        (Q @ x_ref[:, :N]).reshape(-1, order="F"),
        (Qend @ x_ref[:, N]).reshape(-1, order="F"),
        (R @ u_ref).reshape(-1, order="F"),
    )
)
```

This is mathematically equivalent to

```python
cqp = -H @ w_ref
```

but avoids constructing and multiplying the full horizon-sized reference vector.

---

# 4. From dynamics to equality constraints

The discrete dynamics are

$$
x_{k+1}=Ax_k+Bu_k.
$$

Rearranging,

$$
x_{k+1}-Ax_k-Bu_k=0.
$$

Because \(x_0\) is included in the decision vector, the first equality constraint is simply

$$
x_0=\hat{x}_0.
$$

The complete equality system is therefore

$$
A_{\mathrm{qp}}w=b_{\mathrm{qp}}.
$$

The state portion has the block structure

$$
C_x=
\begin{bmatrix}
I & 0 & 0 & \cdots & 0\\
-A&I&0&\cdots&0\\
0&-A&I&\cdots&0\\
\vdots&&\ddots&\ddots&\vdots\\
0&\cdots&0&-A&I
\end{bmatrix}.
$$

The input portion is

$$
C_u=
\begin{bmatrix}
0&0&0&\cdots&0\\
-B&0&0&\cdots&0\\
0&-B&0&\cdots&0\\
\vdots&&\ddots&\ddots&\vdots\\
0&\cdots&0&-B
\end{bmatrix}.
$$

Therefore

$$
\boxed{
A_{\mathrm{qp}}=
\begin{bmatrix}
C_x & C_u
\end{bmatrix}.
}
$$

The right-hand side is

$$
\boxed{
b_{\mathrm{qp}}=
\begin{bmatrix}
\hat{x}_0\\
0\\
\vdots\\
0
\end{bmatrix}.
}
$$

The first \(n_x\) rows enforce the measured initial condition.

Every subsequent block enforces one discrete-time dynamics equation.

---

# 5. Why the equality matrix is sparse

Even when \(A\) and \(B\) are dense, they only appear in a small number of block locations.

For example,

$$
C_x=
\begin{bmatrix}
I & 0 & 0 & 0\\
-A&I&0&0\\
0&-A&I&0\\
0&0&-A&I
\end{bmatrix}.
$$

The matrix is therefore **block-banded**, not dense.

The implementation constructs this structure directly with SciPy sparse matrices:

```python
Cx = (
    sparse.eye(
        (N + 1) * nx,
        format="csc",
    )
    - sparse.kron(
        sparse.diags(
            np.ones(N),
            offsets=-1,
            shape=(N + 1, N + 1),
            format="csc",
        ),
        A,
        format="csc",
    )
)
```

and

```python
Cu = sparse.vstack(
    (
        sparse.csc_matrix((nx, N * nu)),
        sparse.kron(
            sparse.eye(N, format="csc"),
            -B,
            format="csc",
        ),
    ),
    format="csc",
)
```

Finally,

```python
Aqp = sparse.hstack(
    (Cx, Cu),
    format="csc",
)
```

constructs the complete sparse equality matrix.

---

# 6. State and input bounds

The individual box constraints are

$$
x_{\mathrm{lb}}\le x_k\le x_{\mathrm{ub}}
$$

and

$$
u_{\mathrm{lb}}\le u_k\le u_{\mathrm{ub}}.
$$

These are represented directly as QP variable bounds rather than by introducing additional inequality rows.

Thus

$$
\boxed{
w_{\mathrm{lb}}\le w\le w_{\mathrm{ub}}.
}
$$

The state bounds are repeated for all \(N+1\) predicted states:

$$
w_{\mathrm{lb}}^x=
\begin{bmatrix}
x_{\mathrm{lb}}\\
\vdots\\
x_{\mathrm{lb}}
\end{bmatrix},
$$

and the input bounds are repeated for all \(N\) predicted inputs:

$$
w_{\mathrm{lb}}^u=
\begin{bmatrix}
u_{\mathrm{lb}}\\
\vdots\\
u_{\mathrm{lb}}
\end{bmatrix}.
$$

The complete lower bound is

$$
w_{\mathrm{lb}}=
\begin{bmatrix}
w_{\mathrm{lb}}^x\\
w_{\mathrm{lb}}^u
\end{bmatrix},
$$

with an analogous construction for \(w_{\mathrm{ub}}\).

Because \(x_0\) is a decision variable, the state bounds also apply to \(x_0\). Consequently, the current measured state must lie within the specified state bounds for the QP to be feasible.

---

# 7. Complete QP

The complete lifted MPC problem produced by `generate_QP()` is therefore

$$
\boxed{
\begin{aligned}
\min_w\quad&
\frac12w^THw+c^Tw\\
\text{s.t.}\quad&
A_{\mathrm{qp}}w=b_{\mathrm{qp}},\\
&
w_{\mathrm{lb}}\le w\le w_{\mathrm{ub}}.
\end{aligned}
}
$$

where

$$
H=
\operatorname{blkdiag}
\left(
Q,\ldots,Q,Q_{\mathrm{end}},
R,\ldots,R
\right)
$$

and

$$
c=-Hw_{\mathrm{ref}}.
$$

In the implementation, the QP is returned as:

```python
{
    "H": Hqp,
    "c": cqp,
    "A": Aqp,
    "b": bqp,
    "lb": w_lb,
    "ub": w_ub,
}
```

The name `"H"` is used by this implementation for the QP quadratic matrix. When passing it to `qpsolvers`, it corresponds to the library's `P` argument.

Thus:

```python
solve_qp(
    P=qp["H"],
    q=qp["c"],
    A=qp["A"],
    b=qp["b"],
    lb=qp["lb"],
    ub=qp["ub"],
    solver="osqp",
)
```

matches the mathematical problem above.

The standard `qpsolvers` formulation is

$$
\frac12x^TPx+q^Tx
$$

subject to equality constraints, inequality constraints, and variable bounds. The library accepts SciPy CSC matrices for sparse `P` and `A`. Solver-specific keyword arguments can also be passed through `solve_qp`.

---

# 8. Variable ordering and unpacking the solution

The optimizer returns one vector

```text
w = [x0, x1, ..., xN, u0, ..., u(N-1)]
```

where each state and input is a vector.

The state portion occupies

```python
(n + 1) * nx
```

elements.

The input portion occupies

```python
n * nu
```

elements.

For a solution `w`, the trajectories can be recovered as:

```python
x = w[:(N + 1) * nx]
u = w[(N + 1) * nx:]

x = x.reshape((N + 1, nx))
u = u.reshape((N, nu))
```

After reshaping,

```text
x[k, :]
```

is \(x_k^T\), and

```text
u[k, :]
```

is \(u_k^T\).

---

# 9. Data interface

All numerical inputs are NumPy arrays with `float64` precision.

The required dimensions are:

| Parameter | Shape       |
| --------- | ----------- |
| `A`       | `(nx, nx)`  |
| `B`       | `(nx, nu)`  |
| `Q`       | `(nx, nx)`  |
| `R`       | `(nu, nu)`  |
| `Qend`    | `(nx, nx)`  |
| `x0`      | `(nx, 1)`   |
| `x_ref`   | `(nx, N+1)` |
| `u_ref`   | `(nu, N)`   |
| `x_lb`    | `(nx, 1)`   |
| `x_ub`    | `(nx, 1)`   |
| `u_lb`    | `(nu, 1)`   |
| `u_ub`    | `(nu, 1)`   |

where

```text
N  = intervals
nx = number of states
nu = number of inputs
```

---

# 10. Reference trajectory layout

The state reference is stored as

```python
x_ref.shape == (nx, N + 1)
```

where

```text
x_ref[:, 0] -> x0 reference
x_ref[:, 1] -> x1 reference
...
x_ref[:, N] -> xN reference
```

The input reference is stored as

```python
u_ref.shape == (nu, N)
```

where

```text
u_ref[:, 0] -> u0 reference
...
u_ref[:, N-1] -> u(N-1) reference
```

The state reference includes a reference for \(x_0\), even though \(x_0\) itself is fixed by the initial-state equality constraint.

---

# 11. Matrix sparsity

The implementation preserves sparsity at the horizon level.

The Hessian has the structure

$$
H=
\begin{bmatrix}
Q&0&0&\cdots&0\\
0&Q&0&\cdots&0\\
0&0&Q&\cdots&0\\
\vdots&&&\ddots&\vdots\\
0&0&0&\cdots&Q_{\mathrm{end}}\\
&&&&\\
&&&&R
\end{bmatrix}.
$$

The dynamics matrix is block-banded.

Consequently, the memory required by the lifted formulation grows with the number of horizon blocks rather than storing the full dense horizon matrices.

For dense `A` and `B`, the nonzero blocks themselves are dense, but the overall matrices remain sparse because the zero blocks between them are never explicitly constructed.

The implementation therefore converts the small matrices to sparse form only when embedding them into the large horizon matrices:

```python
A = sparse.csc_matrix(self.A)
B = sparse.csc_matrix(self.B)
Q = sparse.csc_matrix(self.Q)
R = sparse.csc_matrix(self.R)
Qend = sparse.csc_matrix(self.Qend)
```

The large matrices are then constructed directly as CSC matrices.

---

# 12. Why CSC?

The lifted matrices are constructed in SciPy's **Compressed Sparse Column (CSC)** format:

```python
format="csc"
```

This is a standard sparse representation and is directly accepted by qpsolvers for sparse QP matrices.

The objective and constraint vectors remain ordinary NumPy arrays because they do not contain the same large block-level sparsity structure.

The resulting representation is therefore:

```text
P / H      -> scipy.sparse.csc_matrix
A          -> scipy.sparse.csc_matrix

q / c      -> numpy.ndarray
b          -> numpy.ndarray
lb         -> numpy.ndarray
ub         -> numpy.ndarray
```

---

# 13. Validation

The `LinearMPC` model validates the input dimensions before constructing the QP.

It also verifies:

### `Q` and `Qend`

$$
Q=Q^T,
\qquad
Q\succeq0
$$

and

$$
Q_{\mathrm{end}}=Q_{\mathrm{end}}^T,
\qquad
Q_{\mathrm{end}}\succeq0.
$$

### `R`

$$
R=R^T,
\qquad
R\succ0.
$$

### Bounds

Every component must satisfy

$$
x_{\mathrm{lb}}\le x_{\mathrm{ub}}
$$

and

$$
u_{\mathrm{lb}}\le u_{\mathrm{ub}}.
$$

These conditions ensure that the supplied weighting matrices define a convex quadratic tracking objective and that the box constraints are internally consistent.

Note that whether a particular QP backend accepts a positive-semidefinite rather than positive-definite quadratic matrix depends on the selected solver. qpsolvers documents that some backends require definiteness while others support semidefinite problems.

---

# 14. Example

```python
import numpy as np

N = 1000
nx = 12
nu = 4

A = np.array([
    [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
    [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
    [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
    [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
    [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
    [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
    [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
    [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
    [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
    [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846],
], dtype=np.float64)

B = np.array([
    [0.,      -0.0726,  0.,     0.0726],
    [-0.0726,  0.,       0.0726, 0.    ],
    [-0.0152,  0.0152, -0.0152, 0.0152],
    [0.,     -0.0006,   0.,     0.0006],
    [0.0006,   0.,      -0.0006, 0.    ],
    [0.0106,   0.0106,   0.0106, 0.0106],
    [0.,     -1.4512,   0.,     1.4512],
    [-1.4512,  0.,       1.4512, 0.    ],
    [-0.3049,  0.3049, -0.3049, 0.3049],
    [0.,     -0.0236,   0.,     0.0236],
    [0.0236,   0.,      -0.0236, 0.    ],
    [0.2107,   0.2107,   0.2107, 0.2107],
], dtype=np.float64)

Q = np.diag([
    0., 0., 10., 10., 10., 10.,
    0., 0., 0., 5., 5., 5.
])

R = 0.1 * np.eye(nu)
Qend = Q.copy()

x0 = np.zeros((nx, 1))

x_ref = np.zeros((nx, N + 1))
x_ref[2, :] = 1.0

u_ref = np.zeros((nu, N))

x_lb = np.array([
    [-np.pi / 6],
    [-np.pi / 6],
    [-100.],
    [-100.],
    [-100.],
    [-1.],
    [-100.],
    [-100.],
    [-100.],
    [-100.],
    [-100.],
    [-100.],
])

x_ub = np.array([
    [np.pi / 6],
    [np.pi / 6],
    [100.],
    [100.],
    [100.],
    [100.],
    [100.],
    [100.],
    [100.],
    [100.],
    [100.],
    [100.],
])

u0 = 10.5916

u_lb = np.full(
    (nu, 1),
    9.6 - u0,
)

u_ub = np.full(
    (nu, 1),
    13.0 - u0,
)

mpc = LinearMPC(
    intervals=N,
    nx=nx,
    nu=nu,
    A=A,
    B=B,
    Q=Q,
    R=R,
    Qend=Qend,
    x0=x0,
    x_ref=x_ref,
    u_ref=u_ref,
    x_lb=x_lb,
    x_ub=x_ub,
    u_lb=u_lb,
    u_ub=u_ub,
)

qp = mpc.generate_QP()
```

---

# 15. Solving the generated QP

The generated QP can be passed directly to qpsolvers.

For example:

```python
from qpsolvers import solve_qp

w = solve_qp(
    P=qp["H"],
    q=qp["c"],
    A=qp["A"],
    b=qp["b"],
    lb=qp["lb"],
    ub=qp["ub"],
    solver="osqp",
)
```

The solver is selected through the `solver` argument.

Backend-specific solver parameters can be supplied as additional keyword arguments. For example:

```python
w = solve_qp(
    P=qp["H"],
    q=qp["c"],
    A=qp["A"],
    b=qp["b"],
    lb=qp["lb"],
    ub=qp["ub"],
    solver="osqp",
    eps_abs=1e-6,
    eps_rel=1e-6,
    max_iter=10000,
    verbose=True,
)
```

The exact available parameters depend on the selected backend. qpsolvers forwards additional keyword arguments to the underlying solver.

---

# 16. Computational structure

The implementation deliberately avoids constructing dense horizon matrices.

The construction flow is:

```text
                 MPC parameters
                       │
                       ▼
              Pydantic validation
                       │
                       ▼
             Lifted MPC formulation
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    Block-diagonal H          Block-banded A
          │                         │
          └────────────┬────────────┘
                       ▼
                Sparse QP matrices
                       │
                       ▼
                qpsolvers backend
                       │
                       ▼
                  optimal w
```

The important property is that `A` and `B` do **not** need to be sparse themselves.

For dense `A` and `B`, only the individual \(A\) and \(B\) blocks are dense. The large lifted matrix remains sparse because the remaining block locations are zero.

---

# 17. Why the lifted formulation?

An alternative is to eliminate all state variables and formulate the problem only in terms of the inputs.

That produces a condensed QP, but the condensed Hessian and constraint matrices are generally much denser.

The lifted formulation retains the state variables and preserves the block structure:

$$
\begin{bmatrix}
I\\
-A&I\\
&-A&I\\
&&\ddots
\end{bmatrix}.
$$

For long horizons, this structure is particularly valuable because the QP retains sparse matrices even when the original system matrices are dense.

Th
