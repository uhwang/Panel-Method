# cython: language_level=3
"""
Cythonized Vxy and RK4 streamline integrator for the Source Panel Method.
Author: Assistant (adapted for your code) -- ChatGPT
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, atan, atan2, cos, sin, fabs
from cpython cimport bool as cbool
cdef double EPS = 1e-12
cdef double TWO_PI = 2.0 * 3.141592653589793

# define floating point type
ctypedef np.float64_t DTYPE_t

# velocity.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, sqrt

def Vxy(object geom, DTYPE_t xp, DTYPE_t yp):
    """
    Compute velocity (vx, vy) at (xp, yp) using the
    full analytical 2D source panel method formulation.
    """

    cdef np.ndarray[DTYPE_t, ndim=1] X = geom.X
    cdef np.ndarray[DTYPE_t, ndim=1] Y = geom.Y
    cdef np.ndarray[DTYPE_t, ndim=1] phi = geom.phi
    cdef np.ndarray[DTYPE_t, ndim=1] SJ = geom.SJ
    cdef np.ndarray[DTYPE_t, ndim=1] sigma = geom.sigma
    cdef int npan = geom.npan

    cdef double U = geom.U
    cdef double alpha = geom.alpha

    cdef double kx = 0.0
    cdef double ky = 0.0
    cdef double A, B, E, P
    cdef double Cx, Cy, Dx, Dy
    cdef double kx1, kx2, ky1, ky2
    cdef double SJj, phij, Xj, Yj, sigmaj
    cdef int j

    for j in range(npan):
        SJj = SJ[j]
        phij = phi[j]
        Xj = X[j]
        Yj = Y[j]
        sigmaj = sigma[j]

        A = -(xp - Xj) * cos(phij) - (yp - Yj) * sin(phij)
        B = (xp - Xj) * (xp - Xj) + (yp - Yj) * (yp - Yj)
        E = sqrt(B - A * A)
        if E == 0.0:
            continue

        P = SJj * SJj + 2.0 * A * SJj + B

        Cx = -cos(phij)
        Cy = -sin(phij)
        Dx = xp - Xj
        Dy = yp - Yj

        kx1 = (Cx * 0.5) * log(P / B)
        ky1 = (Cy * 0.5) * log(P / B)

        kx2 = ((Dx - Cx * A) / E) * (atan((SJj + A) / E) - atan(A / E))
        ky2 = ((Dy - Cy * A) / E) * (atan((SJj + A) / E) - atan(A / E))

        kx += sigmaj / TWO_PI * (kx1 + kx2)
        ky += sigmaj / TWO_PI * (ky1 + ky2)

    vx = U * cos(alpha) + kx
    vy = U * sin(alpha) + ky

    return vx, vy

# ------------------------
# RK4 streamline integrator (Cython)
# ------------------------
def get_streamlines_rk4(object geom, 
                         double minx, double maxx,
                         double miny, double maxy,
                         int nx, int ny,
                         double dt=0.05,
                         int NMAX=1000):
    """
    Compute streamlines using RK4 integration.
    geom : geometry object (passed to Vxyp)
    Vxyp : callable (geom, x, y) -> (u, v)
    """

    cdef np.ndarray[DTYPE_t, ndim=1] gridy = np.linspace(miny, maxy, ny)
    cdef double startx = minx
    cdef int i, step, count
    cdef double x, y, x2, y2, x3, y3, x4, y4
    cdef double u1, v1, u2, v2, u3, v3, u4, v4, avg_u, avg_v

    # allocate once
    cdef np.ndarray[DTYPE_t, ndim=2] pts = np.empty((NMAX, 2), dtype=np.float64)
    stream_lines = []

    for i in range(ny):
        y = gridy[i]
        x = startx
        count = 0
        pts[0, 0] = x
        pts[0, 1] = y

        for step in range(1, NMAX):
            if x < minx or x > maxx or y < miny or y > maxy:
                break

            # call Python velocity function (cannot cdef this easily)
            u1, v1 = Vxy(geom, x, y)
            x2 = x + u1 * dt * 0.5
            y2 = y + v1 * dt * 0.5
            u2, v2 = Vxy(geom, x2, y2)

            x3 = x + u2 * dt * 0.5
            y3 = y + v2 * dt * 0.5
            u3, v3 = Vxy(geom, x3, y3)

            x4 = x + u3 * dt
            y4 = y + v3 * dt
            u4, v4 = Vxy(geom, x4, y4)

            avg_u = (u1 + 2*u2 + 2*u3 + u4) / 6.0
            avg_v = (v1 + 2*v2 + 2*v3 + v4) / 6.0

            x += avg_u * dt
            y += avg_v * dt

            # stop if flow is nearly stagnant
            if sqrt(avg_u*avg_u + avg_v*avg_v) < 1e-6 * geom.U:
                break
                
            count += 1
            pts[count, 0] = x
            pts[count, 1] = y

        # append only the valid part
        stream_lines.append(pts[:count+1].copy())

    return stream_lines