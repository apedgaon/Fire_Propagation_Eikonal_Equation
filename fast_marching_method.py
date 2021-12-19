import numpy as np
import matplotlib.pyplot as plt
import heapq as hq
import time
import eikonal_update as eik

def node_num(idx, jdx, Nt):
    return idx * Nt + jdx

def node_idx(node_num, Nt):
    jdx = int(node_num % Nt)
    idx = int((node_num - jdx) / Nt)
    return idx, jdx

levels = 6
initial = 10
nlist = []
timelist = []
for level in range(levels):
    L = 1.0
    divs = initial * 2**level
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
    F = np.ones((Nt, Nt))

    #plt.contourf(xg, yg, u)
    #plt.show()

    start = time.time()
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
                u[idx, jdx] = h / F[idx, jdx]
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
            eik.eikonal_quad_eq_solver(nnidx, nnjdx, u, Nt, F, h)
            iter = iter + 1
            cn = (u[nnidx, nnjdx], node_num(nnidx, nnjdx, Nt))
            hq.heappush(narrow, cn)

    end = time.time()
    #print("time = ", end - start)
    #print(iter)
    nlist.append(N**2)
    timelist.append(end - start)
    #plt.contourf(xg, yg, u)
    #plt.show()


print(nlist)
print(timelist)
plt.plot(nlist, timelist, 'b-o')
plt.plot(nlist, [x for x in nlist], 'k--')
plt.plot(nlist, [x*np.log(x) for x in nlist], 'g--')
plt.plot(nlist, [x**2 for x in nlist], 'r--')
plt.xscale("log")
plt.yscale("log")
plt.show()