Linear MPC

This project defines a finite-horizon linear model predictive control
(MPC) problem and converts it into a structured quadratic program (QP).

The implementation uses a discrete-time linear model

$$
x_{k+1} = A x_k + B u_k
$$

where:

$x_k \in \mathbb{R}^{n_x}$ is the state,

$u_k \in \mathbb{R}^{n_u}$ is the control input,

$A \in \mathbb{R}^{n_x \times n_x}$ is the state-transition matrix,

$B \in \mathbb{R}^{n_x \times n_u}$ is the input matrix.

The horizon contains $n$ control moves. States are indexed from $1$ to
$n+1$, while controls are indexed from $1$ to $n$:

$$
x_1, x_2, \ldots, x_{n+1}
$$

$$
u_1, u_2, \ldots, u_n
$$

The first state is the current state $x_1 = x_0$. The remaining states
are predicted using the discrete model.

Discrete-Time Model and ZOH

The matrices $A$ and $B$ represent the discrete-time prediction model.
For a continuous-time linear system

$$
\dot{x}(t) = A_c x(t) + B_c u(t)
$$

the discrete model can be obtained by assuming zero-order hold (ZOH) on
the control input over each sampling interval. The resulting model is

$$
x_{k+1} = A x_k + B u_k.
$$

Under ZOH, $u_k$ is held constant during the sampling interval
associated with the transition from $x_k$ to $x_{k+1}$.

This class operates on the discrete matrices $A$ and $B$; discretization
itself is outside the class.

LMPC Problem

The finite-horizon linear MPC problem uses the following dimensions.

Component                      Symbol            Dimension

Horizon length                    $n$               scalar
State dimension                 $n_x$               scalar
Control dimension               $n_u$               scalar
State transition matrix           $A$     $n_x \times n_x$
Input matrix                      $B$     $n_x \times n_u$
State cost matrix                 $Q$     $n_x \times n_x$
Input cost matrix                 $R$     $n_u \times n_u$
Terminal cost matrix        $Q_{end}$     $n_x \times n_x$
Initial state                   $x_0$       $n_x \times 1$
State reference             $x_{ref}$   $n_x \times (n+1)$
Input reference             $u_{ref}$       $n_u \times n$
State lower bound            $x_{lb}$       $n_x \times 1$
State upper bound            $x_{ub}$       $n_x \times 1$
Input lower bound            $u_{lb}$       $n_u \times 1$
Input upper bound            $u_{ub}$       $n_u \times 1$

The optimization variables are

$$
x_1,\ldots,x_{n+1},u_1,\ldots,u_n.
$$

The objective is

$$
\min
\sum_{k=1}^{n}
\lVert x_k-x_{ref,k}\rVert_Q^2
+
\lVert x_{n+1}-x_{ref,n+1}\rVert_{Q_{end}}^2
+
\sum_{k=1}^{n}
\lVert u_k-u_{ref,k}\rVert_R^2
$$

where the weighted norm is

$$
\lVert z\rVert_Q^2 = z^T Q z.
$$

The dynamics constraints are

$$
x_1 = x_0
$$

and

$$
x_{k+1}=A x_k+B u_k,
\qquad k=1,\ldots,n.
$$

The state and input bounds are

$$
x_{lb}\leq x_k\leq x_{ub},
\qquad k=1,\ldots,n+1
$$

and

$$
u_{lb}\leq u_k\leq u_{ub},
\qquad k=1,\ldots,n.
$$

The matrices $Q$ and $Q_{end}$ are positive semidefinite, while $R$ is
positive definite.

Conversion to a QP

The LMPC problem is converted into a quadratic program by stacking every
predicted state and control input into one decision vector.

The stacked vector is

$$
w =
\begin{bmatrix}
x_1\
x_2\
\vdots\
x_{n+1}\
u_1\
u_2\
\vdots\
u_n
\end{bmatrix}.
$$

Its dimension is

$$
w\in\mathbb{R}^{(n+1)n_x+n n_u}.
$$

The conversion has three main parts:

construct the quadratic cost matrix and linear cost vector;

construct the linear equality constraint matrix and right-hand side;

construct the lower and upper bounds on the stacked decision vector.

A constant term in the expanded objective is ignored because it does not
depend on $w$ and therefore does not change the optimizer.

Cost Construction

For one state stage,

x_k^TQx_k
-2x_{ref,k}^TQx_k
+x_{ref,k}^TQx_{ref,k}.
$$

Similarly, the terminal term is

x_{n+1}^TQ_{end}x_{n+1}
-2x_{ref,n+1}^TQ_{end}x_{n+1}
+\text{constant}
$$

and the input terms are

u_k^TRu_k
-2u_{ref,k}^TRu_k
+\text{constant}.
$$

After stacking the variables, the quadratic part is represented by

\operatorname{blkdiag}
\left(
Q,\ldots,Q,Q_{end},
R,\ldots,R
\right).
$$

There are $n$ copies of $Q$, one copy of $Q_{end}$, and $n$ copies of
$R$.

The linear part is

-2
\begin{bmatrix}
Qx_{ref,1}\
Qx_{ref,2}\
\vdots\
Qx_{ref,n}\
Q_{end}x_{ref,n+1}\
Rx_{ref,u,1}\
\vdots\
Rx_{ref,u,n}
\end{bmatrix}
$$

where $x_{ref,u,k}$ denotes the $k$-th input reference column.

Equivalently, using the same array structure as the implementation,

-2
\begin{bmatrix}
\operatorname{vec}(Qx_{ref,1})\
Q_{end}x_{ref,n+1}\
\operatorname{vec}(Ru_{ref})
\end{bmatrix}.
$$

The objective represented by $H_{QP}$ and $c_{QP}$ is

$$
w^T H_{QP} w + c_{QP}^T w
$$

up to the constant reference-only terms that were discarded.

Linear Equality Construction

The dynamics are written so that every state equation contributes one
block row.

The first block row represents the initial condition:

$$
x_1=x_0.
$$

For $k=1,\ldots,n$, the following block row represents

$$
x_{k+1}-Ax_k-Bu_k=0.
$$

The complete equality constraint is therefore

$$
A_{QP}w=b_{QP}.
$$

The state portion has the structured form

$$
C_x =
\begin{bmatrix}
I & 0 & 0 & \cdots & 0\
-A & I & 0 & \cdots & 0\
0 & -A & I & \cdots & 0\
\vdots & & \ddots & \ddots & \vdots\
0 & \cdots & 0 & -A & I
\end{bmatrix}.
$$

There are $n+1$ state block columns and $n+1$ state block rows. Each
block is $n_x\times n_x$.

The input portion is

$$
C_u =
\begin{bmatrix}
0 & 0 & \cdots & 0\
-B & 0 & \cdots & 0\
0 & -B & \cdots & 0\
\vdots & & \ddots & \vdots\
0 & \cdots & -B
\end{bmatrix}.
$$

The first block row is zero because $x_1=x_0$ does not depend on a
control input. The remaining $n$ block rows contain $-B$ in the column
corresponding to the associated control input.

The complete equality matrix is

$$
A_{QP} =
\begin{bmatrix}
C_x & C_u
\end{bmatrix}.
$$

The right-hand side contains the initial state followed by zeros:

\begin{bmatrix}
x_0\
0\
\vdots\
0
\end{bmatrix}.
$$

Thus, the first equality imposes $x_1=x_0$, while every remaining
equality imposes the discrete-time dynamics.

Simple Bound Construction

The state bounds apply to every predicted state:

$$
x_{lb}\leq x_k\leq x_{ub},
\qquad k=1,\ldots,n+1.
$$

The input bounds apply to every control input:

$$
u_{lb}\leq u_k\leq u_{ub},
\qquad k=1,\ldots,n.
$$

Because the decision vector is stacked in the same order as $w$, the
bounds are stacked in the corresponding order:

\begin{bmatrix}
x_{lb}\
\vdots\
x_{lb}\
u_{lb}\
\vdots\
u_{lb}
\end{bmatrix}
$$

and

\begin{bmatrix}
x_{ub}\
\vdots\
x_{ub}\
u_{ub}\
\vdots\
u_{ub}
\end{bmatrix}.
$$

There are $n+1$ copies of each state bound and $n$ copies of each input
bound.

QP Matrix and Vector Dimensions

The complete QP representation returned by generate_QP() has the
following dimensions.

Component                              Symbol                                  Dimension

Decision vector                           $w$                  $((n+1)n_x+n n_u)\times1$

Quadratic cost                       $H_{QP}$   $((n+1)n_x+n n_u)\times((n+1)n_x+n n_u)$
matrix

Linear cost vector                   $c_{QP}$                  $((n+1)n_x+n n_u)\times1$

Equality matrix                      $A_{QP}$         $((n+1)n_x)\times((n+1)n_x+n n_u)$

Equality RHS                         $b_{QP}$                        $((n+1)n_x)\times1$

Lower bound                         $lb_{QP}$                  $((n+1)n_x+n n_u)\times1$

Upper bound                         $ub_{QP}$                  $((n+1)n_x+n n_u)\times1$

The QP is therefore represented by

$$
\min_w
\quad
w^T H_{QP}w+c_{QP}^Tw
$$

subject to

$$
A_{QP}w=b_{QP}
$$

and

$$
lb_{QP}\leq w\leq ub_{QP}.
$$

The returned dictionary contains these components under the keys:

H: $H_{QP}$

c: $c_{QP}$

A: $A_{QP}$

b: $b_{QP}$

lb: $lb_{QP}$

ub: $ub_{QP}$

Validation Checks

The LinearMPC model validates the problem data before the QP is
generated.

Array Conversion

The model converts the matrix and vector inputs to NumPy arrays with
float64 precision.

Cost Matrix Checks

Q and Qend must:

be symmetric;

be positive semidefinite within the numerical tolerance used by the
class.

R must:

be symmetric;

be positive definite within the numerical tolerance used by the
class.

Dimension Checks

The following dimensions are checked against nx, nu, and
intervals:

A: (nx, nx)

B: (nx, nu)

Q: (nx, nx)

R: (nu, nu)

Qend: (nx, nx)

x0: (nx, 1)

x_ref: (nx, n + 1)

u_ref: (nu, n)

x_lb: (nx, 1)

x_ub: (nx, 1)

u_lb: (nu, 1)

u_ub: (nu, 1)

Bound Checks

The model verifies that

$$
x_{lb}\leq x_{ub}
$$

and

$$
u_{lb}\leq u_{ub}.
$$

An invalid bound produces a validation error before QP construction.

Output Structure

generate_QP() returns a dictionary containing the complete structured
QP representation:

{
    "H":  H_QP,
    "c":  c_QP,
    "A":  A_QP,
    "b":  b_QP,
    "lb": lb_QP,
    "ub": ub_QP,
}

This representation keeps the state and input ordering consistent across
the objective, dynamics, and bounds, making the generated QP directly
traceable to the original LMPC formulation.
