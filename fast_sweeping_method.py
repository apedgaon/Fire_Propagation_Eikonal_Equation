import numpy as np
import matplotlib.pyplot as plt
import time
import eikonal_update as eik

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

plt.contourf(xg, yg, u)
plt.show()

F = np.ones((Nt, Nt))
start = time.time()
for idx in range(1, Nt-1):
    for jdx in range(1, Nt-1):
        eik.eikonal_quad_eq_solver(idx, jdx, u, Nt, F, h)

for idx in range(Nt - 2, 0,-1):
    for jdx in range(1, Nt-1):
        eik.eikonal_quad_eq_solver(idx, jdx, u, Nt, F, h)

for idx in range(Nt - 2, 0,-1):
    for jdx in range(Nt - 2, 0,-1):
        eik.eikonal_quad_eq_solver(idx, jdx, u, Nt, F, h)

for idx in range(1, Nt-1):
    for jdx in range(Nt - 2, 0,-1):
        eik.eikonal_quad_eq_solver(idx, jdx, u, Nt, F, h)

end = time.time()
print("time = ", end - start)
plt.contourf(xg, yg, u)
plt.show()