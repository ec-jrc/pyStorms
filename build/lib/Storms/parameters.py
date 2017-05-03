
# coding: utf-8

# ## Definitions Module

# In[1]:

import numpy as np


# In[ ]:

# -------------------------------------------------------------------------
# Const
# -------------------------------------------------------------------------
nm2m=1852. # 1 nautical mile to meters
kt2ms=nm2m/3600.  # knots to m/s
omega=2*np.pi/(3600.*24.) # angular speed omega=2pi*f(=frequency of earth : 1 cycle per day) 2pi* 1 / day in seconds
rhoa=1.15 #air density  Kg/m^3
radius=6378388 #137. # earth's radius according to WGS 84
deg2m=np.pi*radius/180.  # ds on cicle equals ds=r*dth - dth=pi/180
pn=101000.  # Atmospheric pressure [N/m^2] (101KPa - enviromental pressure)

tetaNE=45. #mean angle [degrees] of North Eastern quadrant
tetaNW=135. #        "              North Western
tetaSW=225. #        "              South West
tetaSE=315. #        "              South East

maxR=500.e3  # maximum radius of TC [m] (500Km)


# In[ ]:

kmin=0  # low limit of parameter k (=xn-.5-> k=0-> x=0.5)
kmax=0.25 # upper limit for k (->xn=.65)  WHY?
k0=.1 # initial estimation of holland parameter k

dpmin=10.e2  # minimum value of  pressure drop P_central - P_env(=101kPa).
dpmax=200.e2   # maximum value of  pressure drop P_central - P_env(=101kPa).
dp0=400. #initial estimation of holland parameter dp

rvmaxmin=10.e3  # default minimum value of Rmax[m] 
rmax0=20.e3  # intial estimation for radius of maximum wind [m] (20km)
rmaxmin=5.e3 # intial estimation for radius of maximum wind [m] (5km)

bmin=0.8 # minimum value of holland parameter b
#bmax=2.5
bmax=2.5  # maximum value of holland parameter b
b0=1.2  # initial estimation of holland parameter b

fk=0.92 # coefficient for going from 1m to 10m in velocities
