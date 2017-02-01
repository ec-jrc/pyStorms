import numpy as np
from scipy.optimize import minimize, fmin_slsqp

# -------------------------------------------------------------------------
# Const
# -------------------------------------------------------------------------
nm2m=1852. # 1 nautical mile to meters
kt2ms=nm2m/3600.  # knots to m/s
omega=2*np.pi/(3600.*24.) # angular speed omega=2pi*f(=frequency of earth : 1 cycle per day) 2pi* 1 / day in seconds
rhoa=1.15 #air density  Kg/m^3
radius=6378388 #137. # earth's radius according to WGS 84
deg2m=np.pi*radius/180.  # ds on cicle equals ds=r*dth - dth=pi/180

kmin=0  # low limit of parameter k (=xn-.5-> k=0-> x=0.5)
kmax=0.25 # upper limit for k (->xn=.65)  WHY?
k0=.1 # initial estimation of holland parameter k

dpmin=10.e2  # minimum value of  pressure drop P_central - P_env(=101kPa).
dpmax=200.e2   # maximum value of  pressure drop P_central - P_env(=101kPa).
dp0=400. #initial estimation of holland parameter dp

rvmaxmin=10.e3  # default minimum value of Rmax[m] 
rmax0=20.e3  # intial estimation for radius of maximum wind [m] (20km)
rmaxmin=5.e3 # intial estimation for radius of maximum wind [m] (5km)
maxR=500.e3  # maximum radius of TC [m] (500Km)

bmin=0.8 # minimum value of holland parameter b
#bmax=2.5
bmax=2.5  # maximum value of holland parameter b
b0=1.2  # initial estimation of holland parameter b

# p[0] = B
# p[1] = Rmax
# p[2] = k
# p[3] = dp

def slsqp(R,V,vmax,**kwargs):

    if 'dp' in kwargs.keys():
        dp_=kwargs['dp']
        def func(p,x):
                 return (p[0]/rhoa*(p[1]/x)**p[0]*dp_*np.exp(-(p[1]/x)**p[0]))**(0.5+(x-p[1])/(maxR-p[1])*p[2]) # given dp
        def cf(p,x,y):
                 return p[0]-vmax**2*rhoa*np.exp(1.)/dp_ # given dp
        
        p0=[b0,rmax0,k0] # initial values

        bp=[(bmin,bmax),(rmaxmin,R.min()*.99),(kmin,kmax)]


    else:
        def func(p,x):
                 return (p[0]/rhoa*(p[1]/x)**p[0]*p[3]*np.exp(-(p[1]/x)**p[0]))**(0.5+(x-p[1])/(maxR-p[1])*p[2]) # complete
        def cf(p,x,y):
                 return p[0]-vmax**2*rhoa*np.exp(1.)/p[3] # complete
        
        p0=[b0,rmax0,k0,dp0] # initial values
        bp=[(bmin,bmax),(rmaxmin,R.min()*.99),(kmin,kmax),(dpmin,dpmax)]



    def errf(p,x,y):
            return np.sum((func(p,x)-y)**2)

#       def cf(p,x,y):
#           return vmax-func(p,p[1])


    res1 = minimize(errf, p0, args=(R, V), method='L-BFGS-B', bounds=bp, \
                        options={'disp': True, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 'maxiter': 15000, 'ftol': 2.220446049250313e-09, 'maxcor': 10, 'maxfun': 15000})
#    print ('L-BFGS-B', res1.x)
    res2 = minimize(errf, p0, args=(R, V), method='SLSQP', bounds=bp, tol=1e-3, options={'disp': True, 'iprint': 1, 'eps': 1.4901161193847656e-08, 'maxiter': 100, 'ftol': 1e-06})
#    print ('SLSQP', res2.x, res2.fun, res2.message)
    res = fmin_slsqp(errf, p0, bounds=bp, args=(R, V),f_ieqcons=cf,acc=1e-5)
#    print (res)

    if 'dp' in kwargs.keys():
       b = res[0]
       rmax = res[1]
       k = res[2]
    else:
       b = res[0]
       rmax = res[1]
       k = res[2]
       dp_=res[3]
       
    vmax1 = np.sqrt(b*dp_/(rhoa*np.exp(1)))  # compute estimated vmax

    rmse = np.linalg.norm( func(res,R) - V) / np.sqrt(np.size(V))

    var=[rmse,dp_,b,rmax,k ,np.max(V),vmax1]
    varn=['rmse','dph','b','rmaxh','k','np.max(V)','vmax1']

    dic={el:val for (el,val) in zip(varn,var)}
        

#    print ('Rmax= ', res1.x[1],res2.x[1])
#    print ('F(Rmax)= ', cf(res1.x,0.,0.), cf(res2.x,0.,0.))
    return dic



