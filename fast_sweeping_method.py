import numpy as np
import matplotlib.pyplot as plt
import time
import eikonal_update as eik

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

    #plt.contourf(xg, yg, u)
    #plt.show()

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
    nlist.append(N**2)
    timelist.append(end - start)
    #print("time = ", end - start)
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
plt.xlabel("N (number of grid points)")
plt.ylabel("time (s)")
plt.title("Time complexity of Fast Sweeping Method")
plt.legend(["Fast Sweeping Method", "$\mathcal{O}(N)$", "$\mathcal{O}(N log(N))$", "$\mathcal{O}(N^2)$"])
plt.show()