import matplotlib.pyplot as plt
#!pip install qpsolvers
from qpsolvers import solve_qp
# Exercise 3 LMPC syscop 2023
#mpc parameters
nx = 2
nu = 1
dt=0.1
N = 10
x0=np.array([[np.pi/6],[0]])
#ZOH matrix eponential
A = np.array([[1,0.1],[0.1,0.99]])
B = np.array([[0],[0.1]])
Q = np.eye(nx)
R = np.eye(nu)
#Qend = np.array([[12.25,1.8],[1.8,1.88]]) # incorrect infinite horizon cost in homework
Qend = np.array([[37.18,25.81],[25.81,25.02]])
x_ref = np.zeros((nx, N + 1))
u_ref = np.zeros((nu, N))
x_lb = np.array([[-100],[-100]])
x_ub = np.array([[100],[100]])
u_lb = np.array([[-1]])
u_ub = np.array([[1]])
# plant dynamics
def simulate(x0,u):
  #xnext=x0+dt*np.vstack([x0[1],np.sin(x0[0])-0.1*x0[1]+u*np.cos(x0[1])])
  xnext=A@x0.reshape(nx,1)+B@u.reshape(nu,1)
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
nsim=100
for i in range(nsim):
  umpc=solve_mpc(X[-1])
  xnext=simulate(X[-1],umpc)+np.random.normal(0,0.0)
  X.append(xnext)
  U.append(umpc)
#plot simulation
xsim=np.hstack(X)
tsim=np.linspace(0,nsim*dt,nsim+1)
usim=np.hstack(U).ravel()
plt.rcParams['figure.dpi'] = 200
f,a=plt.subplots(1,2)
f.set_figwidth(10)
f.set_figheight(3)
a[0].plot(tsim,xsim[0,:],label='$x_1$')
a[0].plot(tsim,xsim[1,:],label='$x_2$')
a[0].set_title("State Trajectory")
a[0].set_ylabel('$x_1, x_2$')
a[0].set_xlabel("Time (s)")
a[0].grid(True)
a[0].legend()
a[1].step(tsim[0:-1],usim)
a[1].set_title("Control Input")
a[1].set_ylabel('u')
a[1].set_xlabel("Time (s)")
a[1].grid(True)
