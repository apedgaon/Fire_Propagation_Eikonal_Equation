import numpy as np
import matplotlib.pyplot as plt
import heapq as hq

def node_num(idx, jdx, Nt):
    return idx * Nt + jdx

def node_idx(node_num, Nt):
    jdx = int(node_num % Nt)
    idx = int((node_num - jdx) / Nt)
    return idx, jdx

L = 1.0
divs = 10
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
f = np.sin(np.pi * xg)

alive = []
# for idx in range(Nt):
#     alive.append((0, idx))

for idx in range(0, Nt):
    for jdx in range(0, Nt):
        if (abs(u[idx,jdx] - 0.0) < 1.0e-8):
            alive.append(node_num(idx, jdx, Nt))

print(alive)
hq.heapify(alive)
hq.heappush(alive, 23)
hq.heappush(alive, 90)
hq.heappush(alive, 37)
print(alive)
# for idx in range(Nt):
#     for jdx in range(Nt):
#         u[jdx, idx] = 1
# print(Nt)
# print(node_num(2,7, Nt))
# print(node_idx(209, Nt))

#print(f)
plt.contourf(xg, yg, u)
plt.show()