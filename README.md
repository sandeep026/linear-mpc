## LMPC Problem

The finite-horizon LMPC problem uses the following dimensions.

| Component | Symbol | Code Parameter | Dimension |
| --- | --- | --- | --- |
| Horizon length | $N$ | `intervals` | scalar |
| State dimension | $n_x$ | `nx` | scalar |
| Control dimension | $n_u$ | `nu` | scalar |
| State matrix | $A$ | `A` | $n_x \times n_x$ |
| Input matrix | $B$ | `B` | $n_x \times n_u$ |
| State cost matrix | $Q$ | `Q` | $n_x \times n_x$ |
| Input cost matrix | $R$ | `R` | $n_u \times n_u$ |
| Terminal cost matrix | $Q_{end}$ | `Qend` | $n_x \times n_x$ |
| Initial state | $\bar{x}_0$ | `x0` | $n_x \times 1$ |
| State reference | $x_{ref}$ | `x_ref` | $n_x \times (N+1)$ |
| Input reference | $u_{ref}$ | `u_ref` | $n_u \times N$ |
| State lower bound | $x_{lb}$ | `x_lb` | $n_x \times 1$ |
| State upper bound | $x_{ub}$ | `x_ub` | $n_x \times 1$ |
| Input lower bound | $u_{lb}$ | `u_lb` | $n_u \times 1$ |
| Input upper bound | $u_{ub}$ | `u_ub` | $n_u \times 1$ |

The optimization variables are

$$
x_1,\ldots,x_{N+1},u_1,\ldots,u_N.
$$

The initial state is fixed to

$$
x_1 = \bar{x}_0.
$$

The objective is

$$
\min
\sum_{k=1}^{N}
\|x_k-x_{ref,k}\|_Q^2
+
\|x_{N+1}-x_{ref,N+1}\|_{Q_{end}}^2
+
\sum_{k=1}^{N}
\|u_k-u_{ref,k}\|_R^2.
$$

with weighted norm

$$
\|z\|_Q^2 = z^TQz.
$$

The terminal state uses $Q_{end}$ instead of $Q$.

The dynamics are

$$
x_1 = \bar{x}_0
$$

and

$$
x_{k+1} = A x_k + B u_k,
\qquad k=1,\ldots,N.
$$

The state bounds are

$$
x_{lb} \leq x_k \leq x_{ub},
\qquad k=1,\ldots,N+1
$$

and the input bounds are

$$
u_{lb} \leq u_k \leq u_{ub},
\qquad k=1,\ldots,N.
$$
