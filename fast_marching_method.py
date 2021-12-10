import numpy as np
import matplotlib.pyplot as plt

def node_num(idx, jdx, Nt):
    return idx * Nt + jdx

def node_idx(node_num, Nt):
    jdx = int(node_num % Nt)
    idx = int((node_num - jdx) / Nt)
    return idx, jdx

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
u[0, :] = 1.0
u[-1, :] = 1.0
u[:, 0] = 1.0
u[:, -1] = 1.0
f = np.sin(np.pi * xg)

alive = []
for idx in range(Nt):
    alive.append((0, idx))

print(type(alive[0]))
# for idx in range(Nt):
#     for jdx in range(Nt):
#         u[jdx, idx] = 1
# print(Nt)
# print(node_num(2,7, Nt))
# print(node_idx(209, Nt))

#print(f)
plt.contourf(xg, yg, u)
plt.show()