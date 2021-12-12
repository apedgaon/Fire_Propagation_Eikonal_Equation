import numpy as np
import matplotlib.pyplot as plt
import time

def eikonal_solver(idx, jdx, u, Nt, F):
    if (idx == 0):
        U_H = min(u[1, jdx], 0)
    elif (idx == Nt - 1):
        U_H = min(u[Nt - 2, jdx], 0)
    else:
        U_H = min(u[idx - 1, jdx], u[idx + 1, jdx])

    if (jdx == 0):
        U_V = min(u[idx, 1], 0)
    elif (jdx == Nt - 1):
        U_V = min(u[idx, Nt - 2], 0)
    else:
        U_V = min(u[idx, jdx - 1], u[idx, jdx + 1])

    if (abs(U_H - U_V) > h / F[idx, jdx]):
        u[idx, jdx] = min(U_H, U_V) + h / F[idx, jdx]
    else:
        u[idx, jdx] = 0.5 * ((U_H + U_V) + np.sqrt((U_H + U_V)**2 - 2 * (U_H**2 + U_V**2 - (h/F[idx,jdx])**2)))

L = 1.0
divs = 100
Nt = divs + 1
h = L / divs
N = divs - 1

x = np.linspace(0, L, Nt)
y = np.linspace(0, L, Nt)
xg, yg = np.meshgrid(x, y)

large = 1.0e9
u = large * np.ones([Nt, Nt])
u[0, :] = 0.0
u[-1, :] = 0.0

u[:, 0] = 0.0
u[:, -1] = 0.0

#F = np.ones((Nt, Nt)) * np.exp(-yg)
F = np.ones((Nt, Nt))
#F = 0.05 + np.sin(np.pi * xg) + np.sin(np.pi * yg)
plt.contourf(xg, yg, u)
plt.show()
start = time.time()
nitermax = 4000
for niter in range(0, nitermax):
    utemp = np.copy(u)
    for idx in range(1, Nt-1):
        for jdx in range(1, Nt-1):
            eikonal_solver(idx, jdx, u, Nt, F)

    err = np.linalg.norm(u - utemp, np.inf)
    if (err < 1.0e-8):
        print(niter)
        break

end = time.time()
print("time = ", end - start)

plt.contourf(xg, yg, u)
plt.show()