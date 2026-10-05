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
\end{bmatrix}
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
H_{\mathrm{QP}} = 
\begin{bmatrix}
I_N \otimes Q & 0 & 0 \\
0 & Q_{\mathrm{end}} & 0 \\
0 & 0 & I_N \otimes R
\end{bmatrix}
$$

### Linear cost vector

Define the stacked reference trajectories

$$
\bar{x}^{\mathrm{ref}} = 
\begin{bmatrix}
x_1^{\mathrm{ref}} \\
\vdots \\
x_n^{\mathrm{ref}}
\end{bmatrix},
\qquad
\bar{u}^{\mathrm{ref}} = 
\begin{bmatrix}
u_1^{\mathrm{ref}} \\
\vdots \\
u_n^{\mathrm{ref}}
\end{bmatrix}
$$

Then

$$
c_{\mathrm{QP}} = -
\begin{bmatrix}
(I_N \otimes Q) \bar{x}^{\mathrm{ref}} \\
Q_{\mathrm{end}} x_{N+1}^{\mathrm{ref}} \\
(I_N \otimes R) \bar{u}^{\mathrm{ref}}
\end{bmatrix}
$$

### Equality constraint matrix

Let 

$$
S_n \in \mathbb{R}^{(N+1) \times (N+1)}
$$ 

denote the lower-shift matrix,

$$
S_N = 
\begin{bmatrix}
0 & 0 & \cdots & 0 & 0 \\
1 & 0 & \cdots & 0 & 0 \\
0 & 1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & 1 & 0
\end{bmatrix}
$$

where $e_k$ is the $k$-th canonical basis vector.

Then the state-dynamics matrix is

$$
C_x = I_{N+1} \otimes I_{n_x} - S_N \otimes A
$$

$$
C_x = 
\begin{bmatrix}
I & 0 & 0 & \cdots & 0 \\
-A & I & 0 & \cdots & 0 \\
0 & -A & I & \cdots & 0 \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & -A & I
\end{bmatrix}
$$

The input matrix is

$$
C_u = 
\begin{bmatrix}
0_{n_x \times N n_u} \\
-I_N \otimes B
\end{bmatrix}
$$

$$
C_u = 
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
-B & 0 & \cdots & 0 \\
0 & -B & \cdots & 0 \\
\vdots & \ddots & \ddots & \vdots \\
0 & \cdots & 0 & -B
\end{bmatrix}
$$

Hence,

$$
A_{\mathrm{QP}} = 
\begin{bmatrix}
C_x & C_u
\end{bmatrix}
$$

### Equality-constraint right-hand side

The initial condition is imposed through

$$
b_{\mathrm{QP}} = 
\begin{bmatrix}
x_0 \\
0_{N n_x \times 1}
\end{bmatrix}
$$

Thus the equality constraints compactly represent

$$
x_1 = x_0, \qquad x_{k+1} = A x_k + B u_k, \qquad k = 1, \dots, N.
$$

### Box constraints

Define

$$
\bar{x}_{\mathrm{lb}} = \mathbf{1}_{N+1} \otimes x_{\mathrm{lb}}, \qquad \bar{x}_{\mathrm{ub}} = \mathbf{1}_{N+1} \otimes x_{\mathrm{ub}},
$$

and

$$
\bar{u}_{\mathrm{lb}} = \mathbf{1}_N \otimes u_{\mathrm{lb}}, \qquad \bar{u}_{\mathrm{ub}} = \mathbf{1}_N \otimes u_{\mathrm{ub}}.
$$

Then

$$
w_{\mathrm{lb}} = 
\begin{bmatrix}
\bar{x}_{\mathrm{lb}} \\
\bar{u}_{\mathrm{lb}}
\end{bmatrix},
\qquad
w_{\mathrm{ub}} = 
\begin{bmatrix}
\bar{x}_{\mathrm{ub}} \\
\bar{u}_{\mathrm{ub}}
\end{bmatrix}
$$
