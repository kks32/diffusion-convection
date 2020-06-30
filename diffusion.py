from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from IPython.display import HTML

# box size, m
w = h = 1
# intervals in x-, y- directions, m
dx = dy = 0.01
# Thermal diffusivity, m2.s-1
alphaSoil =  2.08e-7 #2.08e-7 m^2/s
alphaSteel = 1.172e-5 #m^2/s 1.172e-5 m^2/s
alphaWater = 1.39e-7 #m^2/s 1.39e-7 m^2/s
# Porosity
n = 0.4
# Viscosity kg/m
mu = 1.00E-03 
# Permeability m2
k = 1E-13
# Thermal expansion 
beta = 8.80E-05
# Cf
cf = 4290
# rhow
rhow = 980
# gravity
g = 9.81 

# Set conduction to 0 to disable
conduction = 1.

# Temperature of the cable
Tcool, Thot = 0, 70

# Cable geometry, inner radius r, width dr centred at (cx,cy) (mm)
r, cx, cy = 0.02, 0.49, 0.5
r2 = r**2

# Cable geometry, inner radius r, width dr centred at (cx,cy) (mm)
cr1, cx1, cy1 = 0.02, 0.5, 0.45
cr1r2 = cr1**2

# Cable geometry, inner radius r, width dr centred at (cx,cy) (mm)
cr2, cx2, cy2 = 0.02, 0.5, 0.55
cr2r2 = cr2**2

# pipe geometry
pr, pri, px, py = 0.1, 0.06, 0.59, 0.5 #0.59, 0.5
pr2 = pr**2
pri2 = pri**2

# Rayleigh number
#ra = (rhow * g * (h -cx) * k * beta * (Thot-Tcool))/(mu*alphaSoil)
#print("Rayleigh-Darcy number: {}".format(ra))

# Calculations
nx, ny = int(w/dx), int(h/dy)

dx2, dy2 = dx*dx, dy*dy
#dt = dx2 * dy2 / (2 * alphaSoil * (dx2 + dy2))

dt = 0.0001 * 3600
#print("dt: {}, computed dt: {} critical dt: {}".format(dt, dx2 * dy2 / (2 * alphaSoil * (dx2 + dy2)), 1/(2 * alphaSoil * (dx2 + dy2))))

alpha = alphaSoil * np.ones((nx, ny))
u0 = Tcool * np.ones((nx, ny))
u = u0.copy()

uu = []


# Initial conditions
for i in range(nx):
    for j in range(ny):
        #mid cable
        if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2:
            u0[i,j] = Thot
        # left cable
        elif ((i*dx-cx1)**2 + (j*dy-cy1)**2) < cr1r2:
            u0[i,j] = Thot
        # right cable
        elif ((i*dx-cx2)**2 + (j*dy-cy2)**2) < cr2r2:
            u0[i,j] = Thot
        # Modify pipe alpha
        elif ((i*dx-px)**2 + (j*dy-py)**2) < pr2 and ((i*dx-px)**2 + (j*dy-py)**2) > pri2:
            alpha[i,j] = alphaSteel
        # Modify pipe water alpha
        elif ((i*dx-px)**2 + (j*dy-py)**2) < pri2:
            alpha[i,j] = alphaWater
            

        
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit            
def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space, upwind solution for convection
    # Convection velocity term:
    # The velocity corresponds to differential density, since we are measuring the differnetial temp,
    # the rho(1 - beta(T)) is written as rho*(beta*DeltaT)
    '''
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + \
     + conduction * a[1:-1,1:-1] * dt * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2 + \
          (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2) + \
    dt * (1/(n*mu)*k*g*rhow)*(beta*(u0[1:-1,1:-1]-Tcool)) * \
    (u0[2:,1:-1] - u0[1:-1,1:-1])/(dy)
    '''
    for i in range(nx):
        for j in range(ny):
            #mid cable
            if ((i*dx-cx)**2 + (j*dy-cy)**2) < r2:
                u[i,j] = Thot
            # left cable
            elif ((i*dx-cx1)**2 + (j*dy-cy1)**2) < cr1r2:
                u[i,j] = Thot
            # right cable
            elif ((i*dx-cx2)**2 + (j*dy-cy2)**2) < cr2r2:
                u[i,j] = Thot
            elif i > 0 and j > 0 and i < (nx -1) and j < (ny -1):
                # The velocity corresponds to differential density, since we are measuring the differnetial temp,
                # the rho(1 - beta(T)) is written as rho*(beta*DeltaT)
                u[i, j] = u0[i, j] + \
                    + conduction * dt * ((np.minimum(alpha[i, j], alpha[i+1, j]) *(u0[i+1, j] - u0[i,j]) + \
                                          np.minimum(alpha[i, j], alpha[i-1, j]) * (u0[i-1, j] - u0[i, j]))/dy2 + \
                                          (np.minimum(alpha[i, j], alpha[i, j+1]) * (u0[i, j+1] - u0[i,j]) + \
                                           np.minimum(alpha[i, j], alpha[i, j-1]) * (u0[i,j-1] - u0[i,j]))/dx2) + \
                                                        0. * dt * (1/(n*mu)*k*g*rhow)*(beta*(u0[i,j]-Tcool)) * \
                                                        (u0[i+1,j] - u0[i,j])/(dy)
    


    u0 = u.copy()
    return u0, u

# Number of timesteps
nsteps = 50001
npercent = int(nsteps/100)
for m in range(nsteps):
    if m % (npercent) == 0:
        print("Completed: {} %".format(m/npercent))
    u0, u = do_timestep(u0, u)
    if m % (npercent) == 0:
        uu.append(u.copy())

print("Conduction: ", conduction)
print("Total simulation time: {} hours".format(dt * nsteps / 3600))

fig = plt.figure()
pcm = plt.pcolormesh(np.flipud(uu[len(uu)-1]))
plt.colorbar()
plt.show()


#pcm = plt.pcolormesh(np.flipud(uu[0]))
#plt.xlim(40,160)
#plt.ylim(40,200)
#plt.colorbar()    
#plt.show()
