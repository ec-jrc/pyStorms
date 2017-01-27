"Function performing an estimation of the Holland parameters based on Monte Carlo"

import numpy as np

nb=10000  #size of random numbers used

npmin=2 # minimum number of velocities in order to...

kmin=0  # low limit of parameter k (=xn-.5-> k=0-> x=0.5)
kmax=0.25 # upper limit for k (->xn=.65)  WHY?

dpmin=10.e2  # minimum value of  pressure drop P_central - P_env(=101kPa).
dpmax=200.e2   # maximum value of  pressure drop P_central - P_env(=101kPa).
rvmaxmin=10.e3  # default minimum value of Rmax[m] 

bmin=0.8 # minimum value of holland parameter b
#bmax=2.5
bmax=1.8  # maximum value of holland parameter b
b0=1.2  # initial estimation of holland parameter b

rmax0=20.e3  # intial estimation for radius of maximum wind [m] (20km)
maxR=500.e3  # maximum radius of TC [m] (500Km)

# -------------------------------------------------------------------------
# Const
# -------------------------------------------------------------------------
nm2m=1852. # 1 nautical mile to meters
kt2ms=nm2m/3600.  # knots to m/s
omega=2*np.pi/(3600.*24.) # angular speed omega=2pi*f(=frequency of earth : 1 cycle per day) 2pi* 1 / day in seconds
rhoa=1.15 #air density  Kg/m^3
radius=6378388 #137. # earth's radius according to WGS 84
deg2m=np.pi*radius/180.  # ds on cicle equals ds=r*dth - dth=pi/180


def mc(R,V,sinfi,lat,vmax0vt):
    
    npv=np.size(V)

    if (npv>npmin):
        K=kmin+(kmax-kmin)*np.random.rand(nb)
    else:
        K = np.ones(nb)*kmin
     
#  DP
    DP=dpmin*(dpmax/dpmin)**np.random.rand(nb)

    #  Rmax
    rvmaxmin_=np.min([rvmaxmin,np.min(R)*0.5])  # update the minimum  value for Rmax with the R.min/2 from input
    RMAX=rvmaxmin_*(np.min(R)*0.99/rvmaxmin_)**np.random.rand(nb) # range  min(10000,Rmin/2)<Rmax<.99*Rmin (scaled)

    #--------------------------------------------------
    # calculate vmax1 = v max0k -vt - Coriolis effect (function of RMAX)
    #--------------------------------------------------
    deltalatvmax=RMAX/deg2m*sinfi  # for each Rmax we compute the lat deviation for the velocity
    latvmax=lat+deltalatvmax

    fvmax=2*omega*np.abs(np.sin(np.radians(latvmax))) # Coriolis coef f

    fvmax2=RMAX*fvmax/2
    vmax1=((vmax0vt+fvmax2)**2-fvmax2**2)**0.5
    mask=vmax1<np.max(V)
    np.copyto(vmax1,np.max(V),where=mask)

    #----------------------------------
    # use the random values of vmax,dp above we compute b (from Holland 2010 - eqs (7))
    #----------------------------------

    B=(rhoa*np.exp(1)/DP)*vmax1**2


    m=(B >= bmin) & ( B <= bmax) & (lat*latvmax > 0)  # mask B that fits all 3 criteria
    nb1 = np.sum(m) #number of 'True' values

    #  mask arrays accordingly
    K=K[m]
    DP=DP[m]
    RMAX=RMAX[m]
    B = B[m]

    nval = np.size(V)  # number of V > 0
    Vcalc = []
    RMS = np.zeros(nb1)

    #try:
    #        r
    #except NameError:
    #        pass
    #else:
    #        r=None

    # check values for all V
    for i in range(nval):
          try:
            r = R[i]
            ratio=(r-RMAX)/(maxR-RMAX)
            X=0.5 + np.min([np.max(ratio,0),1])*K   #compute x using random k  & Rmax
            Vcalc=np.append(Vcalc,((B/rhoa) * DP* (RMAX/r)**B * np.exp(-(RMAX/r)**B))**X)  # compute & store V
          except: print 'sys.exit()'

    for i in range (nb1):
          try:
            RMS[i]=np.sqrt(np.average((Vcalc[i::nb1]-V)**2))  # compute deviation from estimated and given values
          except : print 'sys.exit()'

    value=nb1
    totvalue=nb

    # -------------------------------------------------------------------------
    # select final velocities
    # -------------------------------------------------------------------------
    m=RMS == np.min(RMS)  #find minimum RMS

    # select the minimizing quantities
    rmse=RMS[m][0]
    dp=DP[m][0]
    b=B[m][0]
    rmax=RMAX[m][0]
    k=K[m][0]

    vmax1 = np.sqrt(b*dp/(rhoa*np.exp(1)))  # compute estimated vmax


    # return
    var=[rmse,dp,b,rmax,k ,np.max(V),vmax1]
    varn=['rmse','dp','b','rmax','k','np.max(V)','vmax1']

    mcdic={el:val for (el,val) in zip(varn,var)}

    return mcdic

