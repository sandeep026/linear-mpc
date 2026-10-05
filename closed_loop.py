from lmpc import LinearMPC
from qpsolver import solve_qp
import matplotlib.pyplot as plt

#mpc parameters
nx = 1
nu = 1
dt=0.1
N = 25
x0=np.array([[2]])
#forward euler
A = np.array([[dt+1]])
B = np.array([[dt]])
Q = np.array([[10]])
R = np.array([[1]])
Qend = np.array([[100]])
x_ref = np.zeros((nx, N + 1))
u_ref = np.zeros((nu, N))
x_lb = np.array([[-5]])
x_ub = np.array([[5]])
u_lb = np.array([[-2.5]])
u_ub = np.array([[2.5]])
# plant dynamics
def simulate(x0,u):
  xnext=A@x0+B@u
  return xnext
# parse and solve
def solve_mpc(x0):
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
  z = solve_qp(
      P=qp["H"],
      q=qp["c"],
      A=qp["A"],
      b=qp["b"],
      lb=qp["lb"],
      ub=qp["ub"],
      solver="clarabel",
      verbose=False,
  )
  # first control input
  return z[nx*(N+1):nx*(N+1)+nu]

#start simulation
X=[x0]
U=[]
nsim=200
for i in range(nsim):
  umpc=solve_mpc(X[-1])
  xnext=simulate(X[-1],umpc)+np.random.normal(0,0.0)
  X.append(xnext)
  U.append(umpc)
#plot simulation
xsim=np.hstack(X).ravel()
tsim=np.linspace(0,nsim*dt,nsim+1)
usim=np.hstack(U).ravel()
plt.rcParams['figure.dpi'] = 200
f,a=plt.subplots(1,2)
a[0].plot(tsim,xsim)
a[0].set_title("State Trajectory")
a[0].set_xlabel("Time (s)")
a[0].grid(True)
a[1].step(tsim[0:-1],usim)
a[1].set_title("Control Input")
a[1].set_xlabel("Time (s)")
a[1].grid(True)
