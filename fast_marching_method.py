import numpy as np
import matplotlib.pyplot as plt
import heapq as hq

def node_num(idx, jdx, Nt):
    return idx * Nt + jdx

def node_idx(node_num, Nt):
    jdx = int(node_num % Nt)
    idx = int((node_num - jdx) / Nt)
    return idx, jdx

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
#f = 1.0 + np.sin(np.pi * xg) + np.sin(np.pi * yg)
f = np.ones((Nt, Nt))

plt.contourf(xg, yg, u)
plt.show()

alive = []
narrow = []
far = []
for idx in range(0, Nt):
    for jdx in range(0, Nt):
        if (abs(u[idx,jdx] - 0.0) < 1.0e-8):
            alive.append((u[idx, jdx], node_num(idx, jdx, Nt)))

for idx in range(1, Nt - 1):
    for jdx in range(1, Nt - 1):
        if (idx == 1 or idx == Nt - 2 or jdx == 1 or jdx == Nt - 2):
            u[idx, jdx] = h / f[idx, jdx]
            narrow.append((u[idx, jdx], node_num(idx, jdx, Nt)))

for idx in range(2, Nt - 2):
    for jdx in range(2, Nt - 2):
        far.append((u[idx, jdx], node_num(idx, jdx, Nt)))

hq.heapify(narrow)
iter = 0
#for iter in range(niter):
while len(narrow):
    c = hq.heappop(narrow)
    alive.append(c)
    nidx, njdx = node_idx(c[1], Nt)
    new_indices = []
    if u[nidx - 1, njdx] == large:
        new_indices.append([nidx - 1, njdx])
    if u[nidx + 1, njdx] == large:
        new_indices.append([nidx + 1, njdx])
    if u[nidx, njdx - 1] == large:
        new_indices.append([nidx, njdx - 1])
    if u[nidx, njdx + 1] == large:
        new_indices.append([nidx, njdx + 1])

    for indices in new_indices:
        nnidx = indices[0]
        nnjdx = indices[1]
        eikonal_solver(nnidx, nnjdx, u, Nt, f)
        iter = iter + 1
        cn = (u[nnidx, nnjdx], node_num(nnidx, nnjdx, Nt))
        hq.heappush(narrow, cn)

print(iter)
plt.contourf(xg, yg, u)
plt.show()