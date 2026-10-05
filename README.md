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

## QP matrices

Define the stacked decision vector as

$$
w
=
\operatorname{col}
\left(
x_1,\ldots,x_{n+1},
u_1,\ldots,u_n
\right).
$$

The QP is

$$
\boxed{
\begin{aligned}
\min_w \quad &
\frac{1}{2}w^\top H_{\mathrm{qp}}w
+
c_{\mathrm{qp}}^\top w
\\
\text{s.t.}\quad &
A_{\mathrm{qp}}w=b_{\mathrm{qp}},
\\
&
w_{\mathrm{lb}}\leq w\leq w_{\mathrm{ub}}.
\end{aligned}
}
$$

### Hessian matrix

The quadratic cost matrix is

$$
\boxed{
H_{\mathrm{qp}}
=
\operatorname{blkdiag}
\left(
I_n\otimes Q,\;
Q_{\mathrm{end}},\;
I_n\otimes R
\right).
}
$$

### Linear cost vector

Define the stacked reference trajectories

$$
\bar{x}^{\mathrm{ref}}
=
\operatorname{col}
\left(
x_1^{\mathrm{ref}},\ldots,x_n^{\mathrm{ref}}
\right),
\qquad
\bar{u}^{\mathrm{ref}}
=
\operatorname{col}
\left(
u_1^{\mathrm{ref}},\ldots,u_n^{\mathrm{ref}}
\right).
$$

Then

$$
\boxed{
c_{\mathrm{qp}}
=
-
\operatorname{col}
\left(
(I_n\otimes Q)\bar{x}^{\mathrm{ref}},
\;
Q_{\mathrm{end}}x_{n+1}^{\mathrm{ref}},
\;
(I_n\otimes R)\bar{u}^{\mathrm{ref}}
\right).
}
$$

### Equality constraint matrix

Let $S_n\in\mathbb{R}^{(n+1)\times(n+1)}$ denote the lower-shift matrix,

$$
S_n
=
\sum_{k=1}^{n}
e_{k+1}e_k^\top,
$$

where $e_k$ is the $k$-th canonical basis vector.

Then the state-dynamics matrix is

$$
\boxed{
C_x
=
I_{n+1}\otimes I_{n_x}
-
S_n\otimes A.
}
$$

The input matrix is

$$
\boxed{
C_u
=
\begin{bmatrix}
0_{n_x\times nn_u}
\\
-I_n\otimes B
\end{bmatrix}.
}
$$

Hence,

$$
\boxed{
A_{\mathrm{qp}}
=
\begin{bmatrix}
C_x & C_u
\end{bmatrix}.
}
$$

### Equality-constraint right-hand side

The initial condition is imposed through

$$
\boxed{
b_{\mathrm{qp}}
=
\operatorname{col}
\left(
x_0,\;
0_{n n_x}
\right).
}
$$

Thus the equality constraints compactly represent

$$
x_1=x_0,
\qquad
x_{k+1}=Ax_k+Bu_k,
\quad k=1,\ldots,n.
$$

### Box constraints

Define

$$
\bar{x}_{\mathrm{lb}}
=
\mathbf{1}_{n+1}\otimes x_{\mathrm{lb}},
\qquad
\bar{x}_{\mathrm{ub}}
=
\mathbf{1}_{n+1}\otimes x_{\mathrm{ub}},
$$

and

$$
\bar{u}_{\mathrm{lb}}
=
\mathbf{1}_{n}\otimes u_{\mathrm{lb}},
\qquad
\bar{u}_{\mathrm{ub}}
=
\mathbf{1}_{n}\otimes u_{\mathrm{ub}}.
$$

Then

$$
\boxed{
w_{\mathrm{lb}}
=
\operatorname{col}
\left(
\bar{x}_{\mathrm{lb}},
\bar{u}_{\mathrm{lb}}
\right),
\qquad
w_{\mathrm{ub}}
=
\operatorname{col}
\left(
\bar{x}_{\mathrm{ub}},
\bar{u}_{\mathrm{ub}}
\right).
}
$$
