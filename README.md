# Linear MPC

A lightweight implementation of **discrete-time linear Model Predictive Control (MPC)** using **NumPy**, **SciPy sparse matrices**, **Pydantic**, and [qpsolvers](https://github.com/qpsolvers/qpsolvers).

The implementation formulates linear MPC directly as a **sparse quadratic program (QP)**, avoiding symbolic computation and preserving sparsity throughout the large horizon-level matrices.

---

## Features

* Discrete-time linear MPC formulation
* NumPy-based numerical inputs
* Sparse QP construction with SciPy
* Supports dense system matrices `A` and `B`
* Sparse/block-diagonal Hessian construction
* Sparse lifted dynamics constraints
* State and input box constraints
* Quadratic tracking cost
* Pydantic-based validation of dimensions and cost matrices
* Compatible with QP solvers supported by `qpsolvers`

The implementation is intended for applications where **memory usage and QP construction time matter**, especially for long MPC horizons.

---

## Mathematical formulation

The controller uses the discrete-time linear system

$$
x_{k+1} = A x_k + B u_k,
$$

where

$$
x_k \in \mathbb{R}^{n_x},
\qquad
u_k \in \mathbb{R}^{n_u}.
$$

For a prediction horizon of \(N\) intervals, the MPC problem is

$$
\begin{aligned}
\min_{\{x_k,u_k\}}
\quad&
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
\qquad k=0,\ldots,N-1.
\end{aligned}
$$

The initial state is fixed to the measured/current state \(x_0\).

---

## Lifted QP formulation

The problem is converted to a single sparse QP.

The decision vector is

$$
w =
\begin{bmatrix}
x_0\\
x_1\\
\vdots\\
x_N\\
u_0\\
\vdots\\
u_{N-1}
\end{bmatrix}.
$$

Its dimension is

$$
n_w = (N+1)n_x + N n_u.
$$

The quadratic cost matrix is block diagonal:

$$
H =
\operatorname{blkdiag}
\left(
\underbrace{Q,\ldots,Q}_{N},
Q_{\mathrm{end}},
\underbrace{R,\ldots,R}_{N}
\right).
$$

Because qpsolvers uses the convention

$$
\frac12 w^T P w + q^T w,
$$

the implementation uses

$$
P=H
$$

for the objective

$$
\frac12(w-w_{\mathrm{ref}})^T H(w-w_{\mathrm{ref}})
$$

up to the constant reference-only term.

The linear term is

$$
q=-H w_{\mathrm{ref}}.
$$

Rather than explicitly forming the entire \(H w_{\mathrm{ref}}\) product, the implementation computes the blocks directly:

```python
cqp = -np.concatenate(
    (
        (self.Q @ self.x_ref[:, :n]).reshape(-1, order="F"),
        (self.Qend @ self.x_ref[:, n]).reshape(-1, order="F"),
        (self.R @ self.u_ref).reshape(-1, order="F"),
    )
)
```

---

## Sparse dynamics constraints

The dynamics are represented as

$$
A_{\mathrm{qp}}w=b_{\mathrm{qp}}.
$$

The state portion of the constraint matrix has the block-banded structure

$$
C_x =
\begin{bmatrix}
I & 0 & 0 & \cdots & 0\\
-A & I & 0 & \cdots & 0\\
0 & -A & I & \cdots & 0\\
\vdots & & \ddots & \ddots & \vdots\\
0 & \cdots & 0 & -A & I
\end{bmatrix}.
$$

The control portion is

$$
C_u =
\begin{bmatrix}
0 & 0 & \cdots & 0\\
-B & 0 & \cdots & 0\\
0 & -B & \cdots & 0\\
\vdots & & \ddots & \vdots\\
0 & \cdots & -B
\end{bmatrix}.
$$

Thus,

$$
A_{\mathrm{qp}}=
\begin{bmatrix}
C_x & C_u
\end{bmatrix}.
$$

Although `A` and `B` may themselves be dense, the lifted MPC constraint matrix is still sparse because only a small number of block locations contain `A` and `B`.

The implementation constructs these matrices directly with SciPy sparse operations:

```python
Cx = (
    sparse.eye(
        (n + 1) * nx,
        format="csc",
    )
    - sparse.kron(
        sparse.diags(
            np.ones(n),
            offsets=-1,
            shape=(n + 1, n + 1),
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
        sparse.csc_matrix((nx, n * nu)),
        sparse.kron(
            sparse.eye(n, format="csc"),
            -B,
            format="csc",
        ),
    ),
    format="csc",
)
```

The resulting matrices are stored in **Compressed Sparse Column (CSC)** format.

---

## Why sparse construction?

A long MPC horizon creates very large lifted matrices.

For example, with

```text
N  = 1000
nx = 12
nu = 4
```

the decision vector contains

$$
(1000+1)12 + 1000(4) = 16012
$$

variables.

A dense Hessian would have approximately

$$
16012^2
$$

entries.

That is unnecessary because the Hessian is block diagonal.

Likewise, the dynamics matrix is block banded rather than dense.

The implementation therefore constructs the large matrices directly as sparse matrices:

```python
Hqp = sparse.block_diag(...)
Aqp = sparse.hstack(...)
```

This avoids creating large dense intermediate matrices and significantly reduces both memory usage and construction time.

---

## Input data

All numerical inputs are converted to `numpy.ndarray` with `float64` precision.

### Dimensions

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
N = intervals
nx = number of states
nu = number of inputs
```

---

## Cost matrix requirements

`Q` and `Qend` must be symmetric positive semidefinite:

$$
Q=Q^T,\qquad Q\succeq0
$$

$$
Q_{\mathrm{end}}=Q_{\mathrm{end}}^T,
\qquad
Q_{\mathrm{end}}\succeq0.
$$

`R` must be symmetric positive definite:

$$
R=R^T,\qquad R\succ0.
$$

These conditions are checked during model validation using the eigenvalues of the matrices.

---

## Installation

Install the required packages with:

```bash
pip install numpy scipy pydantic qpsolvers
```

Then install the solver backend you want to use.

For example, for OSQP:

```bash
pip install osqp
```

The available qpsolvers backends can be inspected with:

```python
import qpsolvers

print(qpsolvers.available_solvers)
```

---

## Example

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
```

Generate the sparse QP:

```python
qp = mpc.generate_QP()
```

Inspect the resulting matrices:

```python
print(qp["H"])
print(qp["A"])

print("H shape:", qp["H"].shape)
print("H nonzeros:", qp["H"].nnz)

print("A shape:", qp["A"].shape)
print("A nonzeros:", qp["A"].nnz)
```

---

## QP output

`generate_QP()` returns:

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

where:

| Key  | Description                       |
| ---- | --------------------------------- |
| `H`  | Sparse quadratic matrix           |
| `c`  | Dense linear cost vector          |
| `A`  | Sparse equality constraint matrix |
| `b`  | Dense equality RHS                |
| `lb` | Dense lower bounds                |
| `ub` | Dense upper bounds                |

The large matrices are stored as SciPy `csc_matrix`.

---

## Solving with qpsolvers

The problem can be passed to a qpsolvers backend using its standard interface.

For a solver such as OSQP:

```python
from qpsolvers import solve_qp

result = solve_qp(
    P=qp["H"],
    q=qp["c"],
    A=qp["A"],
    b=qp["b"],
    lb=qp["lb"],
    ub=qp["ub"],
    solver="osqp",
)
```

The returned result is the optimal decision vector

```python
w = result
```

with ordering

```text
w =
[x0, x1, ..., xN, u0, ..., u(N-1)]
```

The state and control trajectories can therefore be recovered with:

```python
x = w[:(N + 1) * nx]
u = w[(N + 1) * nx:]

x = x.reshape((N + 1, nx), order="C")
u = u.reshape((N, nu), order="C")
```

---

## Sparsity considerations

The implementation intentionally distinguishes between the small model matrices and the large lifted matrices.

`A` and `B` may be dense:

```text
A : nx × nx
B : nx × nu
```

This does not make the lifted MPC QP dense.

The large matrices have structured sparsity:

```text
H:
┌────┐
│ Q  │
├────┤
│    │
│ Q  │
├────┤
│    │
│ Q  │
├────┤
│ Qf │
├────┤
│ R  │
├────┤
│    │
│ R  │
└────┘
```

and

```text
Aqp:

[I    0    0    0   ...]
[-A   I    0    0   ...]
[0   -A    I    0   ...]
[0    0   -A    I   ...]
...
```

Consequently, the number of nonzeros grows approximately linearly with the horizon.

This is especially important for long horizons, where a dense lifted representation becomes prohibitively expensive.

---

## Reference ordering

`x_ref` is expected to have shape

```python
(nx, N + 1)
```

and contains

```text
x_ref[:, 0]   -> reference for x0
x_ref[:, 1]   -> reference for x1
...
x_ref[:, N]   -> reference for xN
```

`u_ref` has shape

```python
(nu, N)
```

and contains

```text
u_ref[:, 0]   -> reference for u0
...
u_ref[:, N-1] -> reference for u(N-1)
```

The objective vector construction uses Fortran-order flattening:

```python
.reshape(-1, order="F")
```

so that each time step remains grouped by state/input dimension consistently with the lifted matrix construction.

---

## Constraints

The implementation supports independent lower and upper bounds for every state and input component.

State constraints:

$$
x_{\mathrm{lb},i}
\le
x_{k,i}
\le
x_{\mathrm{ub},i}.
$$

Input constraints:

$$
u_{\mathrm{lb},i}
\le
u_{k,i}
\le
u_{\mathrm{ub},i}.
$$

The bounds are expanded across the horizon using NumPy:

```python
np.tile(self.x_lb[:, 0], n + 1)
np.tile(self.u_lb[:, 0], n)
```

and therefore do not require explicitly constructing additional inequality matrices.

---

## Validation

The Pydantic model validates:

1. Matrix dimensions
2. Symmetry of `Q`, `Qend`, and `R`
3. Positive semidefiniteness of `Q` and `Qend`
4. Positive definiteness of `R`
5. Lower/upper bound consistency

For example:

```python
if np.any(self.x_lb > self.x_ub):
    raise ValueError(
        "lower must be <= upper bound for x"
    )
```

and similarly for the input bounds.

---

## Performance considerations

The main design goal is to avoid constructing dense horizon-sized matrices.

The following operations remain sparse:

```python
sparse.block_diag(...)
sparse.kron(...)
sparse.vstack(...)
sparse.hstack(...)
```

The resulting matrices are explicitly created in CSC format:

```python
format="csc"
```

This avoids an intermediate dense representation of the lifted QP.

The linear term is also computed without forming the full reference vector and multiplying it by the large Hessian:

```python
cqp = -np.concatenate(
    (
        (self.Q @ self.x_ref[:, :n]).reshape(-1, order="F"),
        (self.Qend @ self.x_ref[:, n]).reshape(-1, order="F"),
        (self.R @ self.u_ref).reshape(-1, order="F"),
    )
)
```

This is particularly useful for large horizons.

---

## Design philosophy

This implementation intentionally uses:

* **NumPy** for numerical data
* **SciPy sparse** for large QP matrices
* **Pydantic** for input validation
* **qpsolvers** as the solver interface

No symbolic modeling layer is required.

The goal is a direct numerical pipeline:

```text
NumPy MPC parameters
        │
        ▼
Pydantic validation
        │
        ▼
Sparse QP construction
        │
        ▼
SciPy CSC matrices
        │
        ▼
qpsolvers backend
        │
        ▼
Optimal MPC trajectory
```

This makes the implementation suitable for long-horizon linear MPC where sparse structure should be preserved from construction through solution.

---

## Project status

This is a compact numerical implementation of lifted linear MPC. Solver-specific features such as warm starts, solver-native statistics, factorization reuse, and repeated in-place QP updates depend on the selected qpsolvers backend.

For repeated real-time MPC solves, the most important optimization is to avoid rebuilding the fixed QP structure unnecessarily and to use a solver/backend that supports efficient problem updates and warm starts.

---

## License

Add your preferred license here.
For example:

```text
MIT License
```
