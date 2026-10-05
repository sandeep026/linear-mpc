# Linear MPC

A lightweight implementation of **discrete-time linear model predictive control (LMPC)** that formulates the MPC optimization problem as a structured sparse quadratic program.

The implementation uses:

* NumPy for numerical model and MPC data
* SciPy sparse matrices for the lifted QP
* Pydantic for input validation

---

# 1. Linear MPC problem

Consider a continuous-time linear system

$$
\dot{x}(t) = A_c x(t) + B_c u(t)
$$

with state

$$
x(t) \in \mathbb{R}^{n_x}
$$

and input

$$
u(t) \in \mathbb{R}^{n_u}.
$$

The controller uses a discrete-time model obtained from the continuous-time system. With a sampling time \(T_s\), the input is assumed to be held constant over each sampling interval using **zero-order hold (ZOH)**:

$$
u(t) = u_k
$$

for

$$
t \in [kT_s,(k+1)T_s).
$$

The resulting discrete-time model is

$$
x_{k+1} = A x_k + B u_k.
$$

Here,

* \(A \in \mathbb{R}^{n_x \times n_x}\)
* \(B \in \mathbb{R}^{n_x \times n_u}\)

and \(x_k\) and \(u_k\) denote the state and input at sampling instant \(k\).

---

# 2. Prediction horizon and indexing

Let the prediction horizon contain \(N\) control intervals.

The predicted states are indexed as

$$
x_1,x_2,\ldots,x_{N+1}.
$$

The predicted control inputs are indexed as

$$
u_1,u_2,\ldots,u_N.
$$

The first predicted state is the current measured state:

$$
x_1 = x_0.
$$

The dynamics are therefore

$$
x_{k+1}=Ax_k+Bu_k,
\qquad
k=1,\ldots,N.
$$

Thus the final predicted state is \(x_{N+1}\).

This indexing gives the following interpretation:

| Quantity      |            Dimension | Meaning                                                  |
| ------------- | -------------------: | -------------------------------------------------------- |
| \(x_k\)       |     \(n_x \times 1\) | State at prediction step \(k\), \(k=1,\ldots,N+1\)       |
| \(u_k\)       |     \(n_u \times 1\) | Control input at prediction step \(k\), \(k=1,\ldots,N\) |
| \(x_0\)       |     \(n_x \times 1\) | Measured/current state                                   |
| \(x_k^{ref}\) |     \(n_x \times 1\) | State reference at step \(k\)                            |
| \(u_k^{ref}\) |     \(n_u \times 1\) | Input reference at step \(k\)                            |
| \(A\)         |   \(n_x \times n_x\) | Discrete-time state matrix                               |
| \(B\)         |   \(n_x \times n_u\) | Discrete-time input matrix                               |
| \(Q\)         |   \(n_x \times n_x\) | Stage-state weighting matrix                             |
| \(Q_{end}\)   |   \(n_x \times n_x\) | Terminal-state weighting matrix                          |
| \(R\)         |   \(n_u \times n_u\) | Input weighting matrix                                   |
| \(x_{lb}\)    |     \(n_x \times 1\) | State lower bound                                        |
| \(x_{ub}\)    |     \(n_x \times 1\) | State upper bound                                        |
| \(u_{lb}\)    |     \(n_u \times 1\) | Input lower bound                                        |
| \(u_{ub}\)    |     \(n_u \times 1\) | Input upper bound                                        |
| \(x_{ref}\)   | \(n_x \times (N+1)\) | Complete state reference trajectory                      |
| \(u_{ref}\)   |     \(n_u \times N\) | Complete input reference trajectory                      |

The state reference contains

$$
x_{ref}
=
[x_1^{ref},\ldots,x_{N+1}^{ref}]
$$

and the input reference contains

$$
u_{ref}
=
[u_1^{ref},\ldots,u_N^{ref}].
$$

---

# 3. Weighted norms

For a vector \(v\) and positive semidefinite matrix \(W\), define the weighted squared norm as

$$
\|v\|_W^2 = v^T W v.
$$

Using this notation makes the MPC objective compact and makes the role of each weighting matrix clear.

---

# 4. Discrete linear MPC problem

The LMPC problem is

$$
\begin{aligned}
\min_{\{x_k,u_k\}}
\quad&
\frac{1}{2}
\sum_{k=1}^{N}
\|x_k-x_k^{ref}\|_Q^2
\\
&+
\frac{1}{2}
\|x_{N+1}-x_{N+1}^{ref}\|_{Q_{end}}^2
\\
&+
\frac{1}{2}
\sum_{k=1}^{N}
\|u_k-u_k^{ref}\|_R^2
\end{aligned}
$$

subject to

$$
x_1=x_0
$$

and

$$
x_{k+1}=Ax_k+Bu_k,
\qquad
k=1,\ldots,N.
$$

The state and input constraints are

$$
x_{lb}\le x_k\le x_{ub},
\qquad
k=1,\ldots,N+1
$$

and

$$
u_{lb}\le u_k\le u_{ub},
\qquad
k=1,\ldots,N.
$$

The first state \(x_1\) is therefore fixed to the current measured state \(x_0\), while \(x_2,\ldots,x_{N+1}\) are predicted states.

---

# 5. Decision vector

All predicted states and control inputs are stacked into one decision vector:

$$
w =
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_N\\
x_{N+1}\\
u_1\\
u_2\\
\vdots\\
u_N
\end{bmatrix}.
$$

The dimensions are

$$
w \in \mathbb{R}^{n_w}
$$

with

$$
\boxed{
n_w=(N+1)n_x+Nn_u.
}
$$

The corresponding reference vector is

$$
w_{ref} =
\begin{bmatrix}
x_1^{ref}\\
x_2^{ref}\\
\vdots\\
x_N^{ref}\\
x_{N+1}^{ref}\\
u_1^{ref}\\
u_2^{ref}\\
\vdots\\
u_N^{ref}
\end{bmatrix}.
$$

---

# 6. Conversion of the cost to a QP

The first step is to collect all weighting matrices into one block-diagonal matrix:

$$
H_{QP}
=
\operatorname{blkdiag}
\left(
\underbrace{Q,\ldots,Q}_{N},
Q_{end},
\underbrace{R,\ldots,R}_{N}
\right).
$$

Its dimension is

$$
\boxed{
H_{QP}
\in
\mathbb{R}^{n_w\times n_w}.
}
$$

Using the decision vector and reference vector, the complete tracking cost can be written as

$$
J
=
\frac{1}{2}
(w-w_{ref})^T
H_{QP}
(w-w_{ref}).
$$

Expanding the quadratic gives

$$
J
=
\frac{1}{2}w^T H_{QP} w
-
w_{ref}^T H_{QP}w
+
\frac{1}{2}
w_{ref}^T H_{QP}w_{ref}.
$$

The final term does not depend on the optimization variable \(w\). It is therefore a constant and can be ignored when solving the optimization problem.

The objective can consequently be written as

$$
\boxed{
J
=
\frac{1}{2}w^T H_{QP}w
+
c_{QP}^T w
}
$$

where

$$
\boxed{
c_{QP}=-H_{QP}w_{ref}.
}
$$

Because \(H_{QP}\) is block diagonal, the linear term can also be constructed block by block:

$$
c_{QP}
=
-
\begin{bmatrix}
Qx_1^{ref}\\
\vdots\\
Qx_N^{ref}\\
Q_{end}x_{N+1}^{ref}\\
Ru_1^{ref}\\
\vdots\\
Ru_N^{ref}
\end{bmatrix}.
$$

This is the form used by the implementation.

---

# 7. Conversion of the dynamics to equality constraints

The discrete dynamics are

$$
x_{k+1}-Ax_k-Bu_k=0.
$$

The first constraint fixes the current state:

$$
x_1=x_0.
$$

The next constraints are

$$
x_2-Ax_1-Bu_1=0,
$$

$$
x_3-Ax_2-Bu_2=0,
$$

and so on until

$$
x_{N+1}-Ax_N-Bu_N=0.
$$

These equations are stacked into

$$
\boxed{
A_{QP}w=b_{QP}.
}
$$

---

## 7.1 State part of the equality matrix

The state part is

$$
C_x =
\begin{bmatrix}
I & 0 & 0 & \cdots & 0\\
-A & I & 0 & \cdots & 0\\
0 & -A & I & \cdots & 0\\
\vdots & \vdots & \ddots & \ddots & \vdots\\
0 & 0 & \cdots & -A & I
\end{bmatrix}.
$$

There are \(N+1\) state block columns and \(N+1\) state block rows.

Therefore

$$
\boxed{
C_x
\in
\mathbb{R}^{(N+1)n_x \times (N+1)n_x}.
}
$$

---

## 7.2 Input part of the equality matrix

The input matrix contains one \(-B\) block for each control:

$$
C_u =
\begin{bmatrix}
0 & 0 & 0 & \cdots & 0\\
-B & 0 & 0 & \cdots & 0\\
0 & -B & 0 & \cdots & 0\\
\vdots & \vdots & \ddots & \ddots & \vdots\\
0 & 0 & \cdots & -B & 0
\end{bmatrix}.
$$

The first block row is zero because the initial-state equation

$$
x_1=x_0
$$

does not contain a control input.

There are \(N+1\) block rows and \(N\) input block columns, so

$$
\boxed{
C_u
\in
\mathbb{R}^{(N+1)n_x \times Nn_u}.
}
$$

---

## 7.3 Complete equality matrix

The complete matrix is

$$
\boxed{
A_{QP}
=
\begin{bmatrix}
C_x & C_u
\end{bmatrix}.
}
$$

Therefore

$$
\boxed{
A_{QP}
\in
\mathbb{R}^{(N+1)n_x
\times
\left((N+1)n_x+Nn_u\right)}.
}
$$

The equality right-hand side is

$$
b_{QP}
=
\begin{bmatrix}
x_0\\
0\\
\vdots\\
0
\end{bmatrix}
$$

with dimension

$$
\boxed{
b_{QP}
\in
\mathbb{R}^{(N+1)n_x}.
}
$$

The first \(n_x\) elements contain the current measured state \(x_0\). All remaining elements are zero.

---

# 8. Conversion of the state and input bounds

The variable bounds are represented directly on the decision vector.

For the states,

$$
x_{lb}\le x_k\le x_{ub},
\qquad
k=1,\ldots,N+1.
$$

For the inputs,

$$
u_{lb}\le u_k\le u_{ub},
\qquad
k=1,\ldots,N.
$$

Therefore the complete lower bound is

$$
w_{lb}
=
\begin{bmatrix}
x_{lb}\\
\vdots\\
x_{lb}\\
u_{lb}\\
\vdots\\
u_{lb}
\end{bmatrix}
$$

and the complete upper bound is

$$
w_{ub}
=
\begin{bmatrix}
x_{ub}\\
\vdots\\
x_{ub}\\
u_{ub}\\
\vdots\\
u_{ub}
\end{bmatrix}.
$$

Their dimensions are

$$
\boxed{
w_{lb},w_{ub}
\in
\mathbb{R}^{n_w}.
}
$$

Explicitly,

$$
w_{lb}
=
\begin{bmatrix}
x_{lb} \\
\vdots \\
x_{lb} \\
u_{lb} \\
\vdots \\
u_{lb}
\end{bmatrix},
\qquad
w_{ub}
=
\begin{bmatrix}
x_{ub} \\
\vdots \\
x_{ub} \\
u_{ub} \\
\vdots \\
u_{ub}
\end{bmatrix},
$$

where the state bounds are repeated \(N+1\) times and the input bounds are repeated \(N\) times.

No additional constraint matrix is required for these simple box constraints.

---

# 9. Complete QP

After stacking the states and inputs, the LMPC problem becomes the following structured QP:

$$
\boxed{
\begin{aligned}
\min_w\quad&
\frac{1}{2}w^T H_{QP} w
+
c_{QP}^T w
\\
\text{s.t.}\quad&
A_{QP}w=b_{QP},
\\
&
w_{lb}\le w\le w_{ub}.
\end{aligned}
}
$$

The QP components have the following dimensions:

| QP component |               Dimension | Description                |
| ------------ | ----------------------: | -------------------------- |
| \(w\)        |        \(n_w \times 1\) | Decision vector            |
| \(H_{QP}\)   |      \(n_w \times n_w\) | Quadratic cost matrix      |
| \(c_{QP}\)   |        \(n_w \times 1\) | Linear cost vector         |
| \(A_{QP}\)   | \((N+1)n_x \times n_w\) | Equality constraint matrix |
| \(b_{QP}\)   |   \((N+1)n_x \times 1\) | Equality constraint vector |
| \(w_{lb}\)   |        \(n_w \times 1\) | Lower bounds               |
| \(w_{ub}\)   |        \(n_w \times 1\) | Upper bounds               |

where

$$
\boxed{
n_w=(N+1)n_x+Nn_u.
}
$$

The matrix \(H_{QP}\) is block diagonal and \(A_{QP}\) is block banded. This structure is what allows the lifted LMPC problem to be represented efficiently as a sparse QP.

---

# 10. Sparse matrix structure

The large QP matrices are constructed as sparse matrices.

The Hessian has the structure

$$
H_{QP}
=
\operatorname{blkdiag}
\left(
Q,\ldots,Q,Q_{end},R,\ldots,R
\right).
$$

Only the diagonal blocks are nonzero.

The equality matrix has the block-banded structure

$$
A_{QP}
=
\begin{bmatrix}
I & 0 & \cdots & 0 & 0 & \cdots & 0\\
-A & I & \cdots & 0 & -B & \cdots & 0\\
0 & -A & \ddots & \vdots & 0 & \ddots & 0\\
\vdots & \vdots & \ddots & I & 0 & \cdots & -B
\end{bmatrix}.
$$

Even when \(A\) and \(B\) are dense matrices, the complete lifted matrix remains sparse because the dense blocks occupy only specific locations.

The large matrices are therefore represented as SciPy CSC matrices, while the vectors remain dense NumPy arrays.

---

# 11. Checks implemented by `LinearMPC`

The class validates the LMPC data before constructing the QP.

## Matrix dimensions

The dimensions of all inputs are checked against `N`, `nx`, and `nu`.

In particular:

$$
A\in\mathbb{R}^{n_x\times n_x}
$$

$$
B\in\mathbb{R}^{n_x\times n_u}
$$

$$
Q,Q_{end}\in\mathbb{R}^{n_x\times n_x}
$$

$$
R\in\mathbb{R}^{n_u\times n_u}
$$

$$
x_0,x_{lb},x_{ub}\in\mathbb{R}^{n_x\times1}
$$

$$
u_{lb},u_{ub}\in\mathbb{R}^{n_u\times1}
$$

$$
x_{ref}\in\mathbb{R}^{n_x\times(N+1)}
$$

$$
u_{ref}\in\mathbb{R}^{n_u\times N}.
$$

An error is raised if any matrix or vector has an unexpected shape.

## Symmetry of the weighting matrices

The class checks that

$$
Q=Q^T
$$

$$
Q_{end}=Q_{end}^T
$$

and

$$
R=R^T.
$$

## Positive semidefiniteness of \(Q\) and \(Q_{end}\)

The class checks that the minimum eigenvalue satisfies

$$
\lambda_{\min}(Q)\geq -10^{-10}
$$

and

$$
\lambda_{\min}(Q_{end})\geq -10^{-10}.
$$

This ensures that the state and terminal cost contributions are convex up to the numerical tolerance used by the validation.

## Positive definiteness of \(R\)

The input weighting matrix is required to be positive definite:

$$
\lambda_{\min}(R)>10^{-10}.
$$

This ensures that the input cost is strictly convex in the control variables.

## Bounds

The class checks that every lower bound is less than or equal to its corresponding upper bound:

$$
x_{lb}\leq x_{ub}
$$

and

$$
u_{lb}\leq u_{ub}.
$$

The QP is constructed only after all of these checks have passed.
