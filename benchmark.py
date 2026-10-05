import numpy as np
import matplotlib.pyplot as plt

Ad = np.array([
    [1,       0,      0, 0, 0, 0, 0.1,     0,      0,  0,      0,      0],
    [0,       1,      0, 0, 0, 0, 0,       0.1,    0,  0,      0,      0],
    [0,       0,      1, 0, 0, 0, 0,       0,      0.1, 0,      0,      0],
    [0.0488,  0,      0, 1, 0, 0, 0.0016,  0,      0,  0.0992, 0,      0],
    [0,      -0.0488, 0, 0, 1, 0, 0,      -0.0016, 0,  0,      0.0992, 0],
    [0,       0,      0, 0, 0, 1, 0,       0,      0,  0,      0,      0.0992],
    [0,       0,      0, 0, 0, 0, 1,       0,      0,  0,      0,      0],
    [0,       0,      0, 0, 0, 0, 0,       1,      0,  0,      0,      0],
    [0,       0,      0, 0, 0, 0, 0,       0,      1,  0,      0,      0],
    [0.9734,  0,      0, 0, 0, 0, 0.0488,  0,      0,  0.9846, 0,      0],
    [0,      -0.9734, 0, 0, 0, 0, 0,      -0.0488, 0,  0,      0.9846, 0],
    [0,       0,      0, 0, 0, 0, 0,       0,      0,  0,      0,      0.9846]
], dtype=np.float64)

Bd = np.array([
    [0,       -0.0726,  0,       0.0726],
    [-0.0726,  0,        0.0726,  0],
    [-0.0152,  0.0152,  -0.0152,  0.0152],
    [0,       -0.0006,  0,       0.0006],
    [0.0006,   0,       -0.0006,   0],
    [0.0106,   0.0106,   0.0106,   0.0106],
    [0,       -1.4512,  0,       1.4512],
    [-1.4512,  0,        1.4512,  0],
    [-0.3049,  0.3049,  -0.3049,   0.3049],
    [0,       -0.0236,  0,       0.0236],
    [0.0236,   0,       -0.0236,   0],
    [0.2107,   0.2107,   0.2107,   0.2107]
], dtype=np.float64)

Q = np.diag([
    0, 0, 10, 10, 10, 10,
    0, 0, 0, 5, 5, 5
])

R = 0.1 * np.eye(4)

Qend = Q.copy()

u0 = 10.5916

u_lb = np.array([
    [9.6 - u0],
    [9.6 - u0],
    [9.6 - u0],
    [9.6 - u0],
], dtype=np.float64)

u_ub = np.array([
    [13 - u0],
    [13 - u0],
    [13 - u0],
    [13 - u0],
], dtype=np.float64)

x_lb = np.array([
    [-np.pi / 6],
    [-np.pi / 6],
    [-100],
    [-100],
    [-100],
    [-1],
    [-100],
    [-100],
    [-100],
    [-100],
    [-100],
    [-100],
], dtype=np.float64)

x_ub = np.array([
    [np.pi / 6],
    [np.pi / 6],
    [100],
    [100],
    [100],
    [100],
    [100],
    [100],
    [100],
    [100],
    [100],
    [100],
], dtype=np.float64)

x0 = np.zeros((12, 1), dtype=np.float64)

xr = np.array([
    [0],
    [0],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
], dtype=np.float64)

from time import perf_counter

timer = []
rounds = 10
intervals = range(50, 1000000, 100000)

for i in intervals:
    t1 = perf_counter()

    for _ in range(rounds):
        x_ref = np.tile(xr, (1, i + 1))
        u_ref = np.zeros((4, i), dtype=np.float64)

        mpc = LinearMPC(
            intervals=i,
            nx=12,
            nu=4,
            A=Ad,
            B=Bd,
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
    t2 = perf_counter()
    timer.append((t2 - t1) / rounds)


N = list(intervals)
plt.figure(figsize=(10,3))
plt.plot(np.hstack(N), np.hstack(timer) * 1e3)
plt.ylabel('Averaged construction time [ms]')
plt.xlabel('Intervals')
plt.grid(True)
