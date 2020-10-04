"""
Solve the start-up flow problem for given initial conditions along with steady-state analytical solutions
Retrieved from: http://ohllab.org/CFD_course/Vivek%2520FDM%2520oscillating%2520boundary.html
Slightly modified.
"""
import numpy as np
import time, sys


# Solve [A][X]=[B]
def CM(N):
    # Build coefficient matrix, [A]
    d = -2 * np.ones(N)  # main diagonal
    d[N - 1] = -1  # last element of main diagonal
    d_n = d.copy()
    l = np.ones(N - 1)  # lower diagonal
    u = np.ones(N - 1)  # upper diagonal

    # Forward elimination of lower-diagonal elements
    for i in range(1, N):
        d_n[i] = d[i] - u[i - 1] * l[i - 1] / d_n[i - 1]

    return l, d_n, u


def TDMA(B, l, d_n, u):
    N = np.size(B)

    # THOMAS ALGORITHM
    # Forward elimination of lower-diagonal elements
    for i in range(1, N):
        B[i] = B[i] - B[i - 1] * l[i - 1] / d_n[i - 1]

    X = np.zeros_like(B)
    # Backward substitution
    X[-1] = B[-1] / d_n[-1]
    for i in range(N - 2, -1, -1):
        X[i] = (B[i] - u[i] * X[i + 1]) / d_n[i]

    return X


def solver(vor, sfn, nt, dt, ny, dy, H):
    t = time.time()  # start timing computation

    vor_n = vor.copy()
    sfn_n = sfn.copy()

    # Create the coefficient maxtrix [A] of order (ny-1)
    l, d_n, u = CM(ny - 1)

    for i in range(1, int(nt + 1)):
        # Time-marhcing for vorticity transport equation
        vor_n[1:-1] = vor[1:-1] + dt * (vor[2:] + vor[:-2] - 2 * vor[1:-1]) / dy / dy

        # Solving for streamfunction at (n+1)th step
        B = -vor_n[1:] * dy * dy
        sfn_n[1:] = TDMA(B, l, d_n, u)
        vor_n[0] = 2 * (H * np.sin(i * dt) - sfn_n[1] / dy) / dy  # vorticity boundary condition at y=0
        vor = vor_n.copy()

    # Solve for computed velocity
    u = np.zeros(ny)
    u[0] = H * np.sin(nt * dt)
    u[1:-1] = (sfn_n[2:] - sfn_n[:-2]) / 2 / dy
    u[-1] = 0

    # Analytical solution for vorticity (steady-state)
    r = np.sqrt(1. / 2.)
    tt = nt * dt  # total time
    vor_a = H * r * np.exp(-r * y) * (np.sin(tt - r * y) + np.cos(tt - r * y))

    # Analytical solution for streamfunction (steady-state)
    sfn_a = H * r * (np.exp(-r * y) * (np.cos(tt - r * y) - np.sin(tt - r * y)) - (np.cos(tt) - np.sin(tt)))

    # Analytical solution for velocity (steady-state)
    u_a = H * np.exp(-r * y) * np.sin(tt - r * y)

    t = time.time() - t  # stop timing computation
    print('Done (' + ('%.2f' % t) + 's)')  # display computation time

    return vor, sfn_n, u, vor_a, sfn_a, u_a


## Solve the start-up flow problem for given initial conditions along with steady-state analytical solutions

def compute_stokes_boundary_layer():
    Y = 15. # domain: x characteristic lengths (must be sufficiently large)
    ny = 201 # no. of grid pts.
    dy = Y/(ny-1) # cell size
    y = np.linspace(0,Y,ny)

    dt = 0.001 # size of time step

    nu = 0.01
    U0 = 1.
    f = 1.
    w = 2*np.pi*f
    H = U0/np.sqrt(nu*w)
    print( 'H = ', ('%.2f' %H))

    CFL = dt/dy/dy
    if CFL <= 0.5:
        print( 'CFL = ', ('%.2f' %CFL), 'is OK.')

    # Initial conditions
    vor_0 = np.zeros(ny)
    sfn_0 = np.zeros(ny)

    ntpc = int(2*np.pi/dt)  # no. of time steps per cycle (2pi)

    vor_1, sfn_1, u_1, vor_1a, sfn_1a, u_1a = solver(vor_0, sfn_0, ntpc/10, dt,ny,dy,H) # solve for 10% of 1st cycle
    vor_2, sfn_2, u_2, vor_2a, sfn_2a, u_2a = solver(vor_0,sfn_0, ntpc/2, dt,ny,dy,H) # solve for 50% of 1st cycle
    vor_3, sfn_3, u_3, vor_3a, sfn_3a, u_3a = solver(vor_0,sfn_0,ntpc,dt,ny,dy,H) # solve for 1st cycle
    vor_4, sfn_4, u_4, vor_4a, sfn_4a, u_4a = solver(vor_3,sfn_3,ntpc,dt,ny,dy,H) # solve for 2nd cycle
    vor_5, sfn_5, u_5, vor_5a, sfn_5a, u_5a = solver(vor_4,sfn_4,3*ntpc,dt,ny,dy,H) # solve for 5th cycle
    vor_6, sfn_6, u_6, vor_6a, sfn_6a, u_6a = solver(vor_5,sfn_5,5*ntpc,dt,ny,dy,H) # solve for 10th cycle


