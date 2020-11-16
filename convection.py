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
# Thermal conductivity W/(m-K)
thermal_conductivity = 0.9
# Thermal diffusivity, m2.s-1
alphaSoil =  2.08e-7 #2.08e-7 m^2/s
alphaSteel = 1.172e-5 #m^2/s 1.172e-5 m^2/s
alphaWater = 1.39e-7 #m^2/s 1.39e-7 m^2/s
# Porosity
n = 0.4
# Viscosity kg/m
mu = 1.00E-03 
# Permeability m2
k = 1E-12
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

# pipe geometry
pr, pri, px, py = 0.1, 0.06, 0.62, 0.5 #0.59, 0.5
pr2 = pr**2
pri2 = pri**2

# Rayleigh number
ra = (rhow * g * (h -cx) * k * beta * (Thot-Tcool))/(mu*alphaSoil)
print("Rayleigh-Darcy number: {}".format(ra))

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

flux = np.zeros((nx, ny))
fluxx = np.zeros((nx, ny))
fluxy = np.zeros((nx, ny))

fflux = []
ffluxx = []
ffluxy = []


# Initial conditions
for i in range(nx):
    for j in range(ny):
        # Modify pipe alpha
        if ((i*dx-px)**2 + (j*dy-py)**2) < pr2 and ((i*dx-px)**2 + (j*dy-py)**2) > pri2:
            alpha[i,j] = alphaSteel
        # Modify pipe water alpha
        elif ((i*dx-px)**2 + (j*dy-py)**2) < pri2:
            alpha[i,j] = alphaWater
        
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit            
def do_timestep(u0, u, flux, fluxx, fluxy):
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
    # Set flux to zero
    flux = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            #pipe
            if ((i*dx-px)**2 + (j*dy-py)**2) < pr2:
                u[i,j] = Thot
            elif i > 0 and j > 0 and i < (nx -1) and j < (ny -1):
                # The velocity corresponds to differential density, since we are measuring the differnetial temp,
                # the rho(1 - beta(T)) is written as rho*(beta*DeltaT)
                u[i, j] = u0[i, j] + \
                    + conduction * dt * ((np.minimum(alpha[i, j], alpha[i+1, j]) *(u0[i+1, j] - u0[i,j]) + \
                                          np.minimum(alpha[i, j], alpha[i-1, j]) * (u0[i-1, j] - u0[i, j]))/dy2 + \
                                          (np.minimum(alpha[i, j], alpha[i, j+1]) * (u0[i, j+1] - u0[i,j]) + \
                                           np.minimum(alpha[i, j], alpha[i, j-1]) * (u0[i,j-1] - u0[i,j]))/dx2) + \
                                                        dt * (1/(n*mu)*k*g*rhow)*(beta*(u0[i,j]-Tcool)) * \
                                                        (u0[i+1,j] - u0[i,j])/(dy)

            #Flux around cable
            # -k * dT/dx
            if (((i*dx-cx)**2 + (j*dy-cy)**2) < (r2 + dx**2) and ((i*dx-cx)**2 + (j*dy-cy)**2) > (r2 - dx**2)) or (((i*dx-cx1)**2 + (j*dy-cy1)**2) < (cr1r2 + dx**2) and ((i*dx-cx1)**2 + (j*dy-cy1)**2) > (cr1r2 - dx**2)) or  (((i*dx-cx2)**2 + (j*dy-cy2)**2) < (cr2r2 + dx**2) and ((i*dx-cx2)**2 + (j*dy-cy2)**2) > (cr2r2 - dx**2)):
                fluxx[i,j] = - thermal_conductivity * (u0[i,j-1] - u0[i, j+1])/(2*dx)
                fluxy[i,j] = - thermal_conductivity * (u0[i-1,j] - u0[i+1, j])/(2*dy)
                flux[i,j] = np.sqrt(fluxx[i,j]**2 + fluxy[i,j]**2)

            
    u0 = u.copy()
    return u0, u, flux, fluxx, fluxy

# Number of timesteps
nsteps = 500001
npercent = int(nsteps/100)
id = 0
fluxt = []
for m in range(nsteps):
    u0, u, flux, fluxx, fluxy = do_timestep(u0, u, flux, fluxx, fluxy)
    if m % (npercent) == 0:
        uu.append(u.copy())
        fflux.append(flux.copy())
        ffluxx.append(fluxx.copy())
        ffluxy.append(fluxy.copy())
        print("Completed: {} % flux: {} u: {}".format(m/npercent, np.sum(flux[47:54, 42:58]), np.sum(uu[id])))
        fluxt.append(np.sum(flux[47:54, 42:58]))
        id = id + 1


print("Conduction: ", conduction)
print("Total simulation time: {} hours".format(dt * nsteps / 3600))

fig = plt.figure()
pcm = plt.pcolormesh(np.flipud(uu[len(uu)-1]))
#plt.xticks([0,40,80,120,160,200],[0,20,40,60,80,100])
#plt.yticks([0,40,80,120,160,200],[0,20,40,60,80,100])
plt.colorbar()
plt.show()

heat = u[:,50]
np.savetxt("heat.csv", heat, delimiter=",")

np.savetxt("flux.csv", fluxt, delimiter=",")

#pcm = plt.pcolormesh(np.flipud(fflux[len(fflux)-1]))
#plt.colorbar()    
#plt.show()
