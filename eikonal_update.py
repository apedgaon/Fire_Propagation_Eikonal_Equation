# updates value of time function by solving quadratic equation of 
# discretized version of eikonal equaiton
import numpy as np

def eikonal_quad_eq_solver(idx, jdx, u, Nt, F, h):
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