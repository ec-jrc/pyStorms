import numpy as np
import datetime
import pandas as pd
import glob
from netCDF4 import Dataset
import re
import xml.etree.ElementTree as et 
from xml.dom import minidom

from parameters import *
from mc import mc
from utils import *

radcols=['64ne', '64se', '64sw', '64nw', '50ne', '50se', '50sw', '50nw',
       '34ne', '34se', '34sw', '34nw']
       
       

class Storm(object):
    """Create a storm object for analysing TC data
    # Create an empty grid
    grid = Grid()
    # Load a grid from file
    grid = Grid.fromfile('filename.grd')
    # Write grid to file
    Grid.write(grid,'filename.grd')
    """
    def __init__(self, **kwargs):
        self.properties = kwargs.get('properties', {})
        self.name = kwargs.get('name', None)
        self.basin     = kwargs.get('basin', None)
        self.year     = kwargs.get('year', None)
        
    
#    def parse(self,url):
        
    def frombt(self):
        
        btfile = self.properties['btfile']
        btpath = self.properties['btpath']
                
        data=pd.read_csv(btfile, header=1, low_memory=False)
        data['Basin'] = data['Basin'].str.strip() # strip spaces        
        
        hurdata = data[data['Name'].str.contains(self.name,na=False) & data['ISO_time'].str.contains(self.year,na=False)].copy() 
        
        ntc=hurdata['Basin'].unique() #store all basins with the same TC name  
        hur1=hurdata.loc[hurdata['Basin']==self.basin].copy()   

        hur1[['Longitude','Latitude']]=hur1.loc[:,['Longitude','Latitude']].apply(pd.to_numeric)
        
        #Check if we cross International Date Line (IDL)
        sig=np.sign(hur1.Longitude.values)
        sig1=sig[0]
        m=sig != sig1
        
        if sum(m)>0:
        # adjust the lon values going from -180:180
                if sig1 > 0:
                        hur1.loc[hur1.Longitude < 0, ['Longitude']] = hur1.Longitude+360.
                elif sig1 < 0:
                        hur1.loc[hur1.Longitude > 0, ['Longitude']] = hur1.Longitude-360.


        idf=hur1.iloc[0].Serial_Num # get id of the strom
        
        fname=glob.glob(btpath+'/{}*'.format(idf)) #parse the folder for the id
        dat=Dataset(fname[0]) # retreive data
        
        wradii=dat['atcf_wrad']
        
        wradiinp=np.array(wradii)
        
        w34= wradiinp[:,0,:]
        
        w34ne=w34[:,0]
        w34nw=w34[:,1]
        w34se=w34[:,2]
        w34sw=w34[:,3]
        
        w50=wradiinp[:,1,:]
        
        w50ne=w50[:,0]
        w50nw=w50[:,1]
        w50se=w50[:,2]
        w50sw=w50[:,3]
        
        w64=wradiinp[:,2,:]
        
        w64ne=w64[:,0]
        w64nw=w64[:,1]
        w64se=w64[:,2]
        w64sw=w64[:,3]
        
        isot=dat['isotime'] # time reference
        
        isot1=[''.join(x) for x in isot]
        
        time=[re.sub(r'[-: ]+','',x)[:-4] for x in isot1]
        
        lat=dat['atcf_lat'][:]
        lon=dat['atcf_lon'][:]
        pcenter=dat['atcf_pres'][:]
        vmaxh=dat['atcf_wind'][:] # 10 minute wind in Knots
        rmaxh=dat['atcf_rmw'][:]*nm2m # convert to m
        
        # put it in a dic
        dic={'t':time, 'lat':lat,'lon':lon,'pcenter':pcenter,'vmax':vmaxh, 'rmax':rmaxh, '34ne': w34ne, '34nw': w34nw, '34se':w34se, '34sw':w34sw,\
                                 '50ne': w50ne, '50nw': w50nw, '50se':w50se, '50sw':w50sw,'64ne': w64ne, '64nw': w64nw, '64se':w64se, '64sw':w64sw} 
          
        inpData=pd.DataFrame(dic) # put it in pandas
        
        inpData=inpData.apply(pd.to_numeric) # convert to float
        
        #Take care of the NaNs
        inpData.loc[inpData['lat'] == 9.969210e+36, 'lat'] = np.nan
        inpData.loc[inpData['lon'] == 9.969210e+36, 'lon'] = np.nan
        inpData.loc[inpData['pcenter'] == -32767., 'pcenter'] = np.nan
        inpData.loc[inpData['rmax'] == -60684484., 'rmax'] = np.nan
        inpData.loc[inpData['vmax'] == -32767, 'vmax'] = np.nan
        inpData.loc[inpData['34ne'] == -32767, '34ne']=np.nan
        inpData.loc[inpData['34nw'] == -32767, '34nw']=np.nan
        inpData.loc[inpData['34se'] == -32767, '34se']=np.nan
        inpData.loc[inpData['34sw'] == -32767, '34sw']=np.nan
        inpData.loc[inpData['50ne'] == -32767, '50ne']=np.nan
        inpData.loc[inpData['50nw'] == -32767, '50nw']=np.nan
        inpData.loc[inpData['50se'] == -32767, '50se']=np.nan
        inpData.loc[inpData['50sw'] == -32767, '50sw']=np.nan
        inpData.loc[inpData['64ne'] == -32767, '64ne']=np.nan
        inpData.loc[inpData['64nw'] == -32767, '64nw']=np.nan
        inpData.loc[inpData['64se'] == -32767, '64se']=np.nan
        inpData.loc[inpData['64sw'] == -32767, '64sw']=np.nan  
        
        # Take care of empty data
        inpData.loc[inpData['34ne'] == -1, '34ne']=0
        inpData.loc[inpData['34nw'] == -1, '34nw']=0
        inpData.loc[inpData['34se'] == -1, '34se']=0
        inpData.loc[inpData['34sw'] == -1, '34sw']=0
        inpData.loc[inpData['50ne'] == -1, '50ne']=0
        inpData.loc[inpData['50nw'] == -1, '50nw']=0
        inpData.loc[inpData['50se'] == -1, '50se']=0
        inpData.loc[inpData['50sw'] == -1, '50sw']=0
        inpData.loc[inpData['64ne'] == -1, '64ne']=0
        inpData.loc[inpData['64nw'] == -1, '64nw']=0
        inpData.loc[inpData['64se'] == -1, '64se']=0
        inpData.loc[inpData['64sw'] == -1, '64sw']=0
            
            
        inpData=inpData.dropna() # drop NaN
        
        
        self.data = inpData
        
    def toSI(self,path='.', pref=pn, output=False):
      
        dph=pref-self.data.pcenter*100. #compute dp from reference pressure
        self.data=self.data.assign(dp=dph)  #add to DataFrame
        
        self.data['t']=pd.to_datetime(self.data['t'],format='%Y%m%d%H')
        
        self.data['time']=self.data['t']-self.data.t.iloc[0] # evaluate dt from t=0
        
        self.data['time']=self.data['time'] / pd.Timedelta('1 hour') # convert to hours difference
        
        
        # convert to SI
        self.data['vmax']=self.data['vmax']*kt2ms 
        
        self.data[['64ne','64se','64sw','64nw','50ne','50se','50sw','50nw','34ne','34se','34sw','34nw']]=\
                self.data[['64ne','64se','64sw','64nw','50ne','50se','50sw','50nw','34ne','34se','34sw','34nw']]*nm2m
          
        #set inpData file        
        column_order=['lat','lon','dp','vmax','64ne','64se','64sw','64nw','50ne','50se','50sw','50nw','34ne','34se','34sw','34nw']        
        
        header=['lat','long','dp','vmax','64ne','64se','64sw','64nw','50ne','50se','50sw','50nw','34ne','34se','34sw','34nw']
        
        if output:
              self.data=self.data.set_index('time') # set index
              self.data.to_csv(path+'/inpData.txt',index=True, columns=column_order, sep='\t', header=header) # save inpData file
              self.data=self.data.reset_index() # reset index
              
        
        #set bulInfo file        
        tt=pd.to_datetime(self.data['t'][0])
        
        tt=datetime.datetime.strftime(tt,'%d %b %Y %H:%M:%S')
        
        dic0={'advNo':[1],'tShift':[0],'$date':tt,'land':[1],'notes':[0]}
        
        bul=pd.DataFrame.from_dict(dic0)
        
        if output:        
              bul.to_csv(path+'/bulInfo.txt',index=False, columns=['advNo','tShift','$date','land','notes'], sep='\t')  
        
        #set info.xml file
        info = et.Element('setexp')
        et.SubElement(info, 'source').text = 'Tropical Cyclone Bulletin through GDACS/PDC'
        et.SubElement(info, 'hurName').text = self.name
        et.SubElement(info, 'hurId').text = self.name
        et.SubElement(info, 'basin').text = self.basin
        et.SubElement(info, 'bulNo').text = '1'
        et.SubElement(info, 'bulDate').text = tt
        et.SubElement(info, 'n').text = '100000'
        et.SubElement(info, 'fk').text = '0.81'
        et.SubElement(info, 'stormsurge').text = '0'
        et.SubElement(info, 'timefactor').text = '1'
        et.SubElement(info, 'landfall').text = '1'     
        
        xmlf = minidom.parseString(prettify(info))
        
        if output:
           with open(path+'/info.xml','w') as f:
              xmlf.writexml(f) 
    
      
    def tranvel(self):
        
        x=self.data.lon
        y=self.data.lat 
        
        dt=np.gradient(self.data.time)*3600 # compute dt (translate time from hours to sec)
        
        dx_dt = np.gradient(x,dt)
        dy_dt = np.gradient(y,dt)
        velocity = np.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
        
        #velocity
        vtrx = velocity[:,0] * deg2m * np.cos(np.radians(self.data.lat.values))  #adjust for latitude
        vtry = velocity[:,1] * deg2m
        
        vtr = np.sqrt(vtrx**2+vtry**2)
        
        #Compute the tangent of unit vector value see http://stackoverflow.com/questions/28269379/curve-curvature-in-numpy
        
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        
        tangent = np.array([1/ds_dt] * 2).transpose() * velocity
        
        phi=np.arctan2(tangent[:,1],tangent[:,0]) # the angle of the velocity vector
        
        cosfi = np.cos(phi)
        sinfi = np.sin(phi)
        
        # extend dataset to save new data
        self.data['vtrx']=vtrx
        self.data['vtry']=vtry
        self.data['vtr']=vtr
        self.data['cosfi']=cosfi
        self.data['sinfi']=sinfi
        
        cols=['w'+ x for x in radcols]
        
        an=np.array([tetaNE, tetaSE, tetaSW, tetaNW,tetaNE, tetaSE, tetaSW, tetaNW,tetaNE, tetaSE, tetaSW, tetaNW])# to be used
        sinan = np.sin(np.radians(an+90))  # an +90 = angle of tangential wind
        cosan=np.cos(np.radians(an+90))
        
        V0=np.array([64, 64, 64, 64, 50, 50, 50, 50, 34, 34, 34, 34])*kt2ms*fk #translate knots to m/s and from 1km to 10km
        
        R=self.data.ix[:,radcols].copy()
        
        R=R[R>0]
        
        RATIO = (rmax0/R)**b0    # assume exponential decay eqs (13) from JRC report
        EXPRATIO = np.exp(-RATIO)  #                       "
        
        VT=vtr[:,np.newaxis]*(cosfi[:,np.newaxis] * cosan + sinfi[:,np.newaxis] * sinan)*(1-EXPRATIO)   # Eq (15) from JRC report
        
        VT.loc[self.data.lat<0] = -VT # reverse for south hemishpere
        
        VV = V0-VT   # substract translational velocity from TC velocity
        
        deltalatWR=R/deg2m*np.sin(np.radians(an))
        
        latWR=self.data.lat[:,np.newaxis]+deltalatWR
        
        fWR=2*omega*np.abs(np.sin(np.radians(latWR))) # Coriolis parameter f=2*Omega*sin(lat)
        Vnco=((VV+R*fWR/2)**2-(R*fWR/2)**2)**0.5
        
        Vnco=Vnco.replace(np.nan,0)
        
        #change header
        Vnco.columns = cols
        
        # extend dataset to save the velocities
        self.data = pd.concat([self.data, Vnco], axis=1)
        
        vs = self.data.vmax*fk-vtr
        
        vmax0vt = np.maximum(vs,Vnco.max(axis=1))
        
        self.data['vmax0vt'] = vmax0vt
        
   #     self.data = self.data.set_index('time')
        
                     
    def writenc(self,filename):

        ni=self.lon.shape[0]
        nj=self.lat.shape[0]

        rootgrp = Dataset(filename, 'w', format='NETCDF3_64BIT')
        lats = rootgrp.createDimension('LAT', nj)
        lons = rootgrp.createDimension('LON', ni)
        time = rootgrp.createDimension('TIME', None)


        longitudes = rootgrp.createVariable('LON','f8',('LON',))
        latitudes = rootgrp.createVariable('LAT','f8',('LAT',))
        u = rootgrp.createVariable('U','f8',('TIME','LAT','LON'))
        v = rootgrp.createVariable('V','f8',('TIME','LAT','LON'))
        times = rootgrp.createVariable('TIME','f8',('TIME',))
        p = rootgrp.createVariable('P','f8',('TIME','LAT','LON'))

        rootgrp.description = ''
        rootgrp.history = 'JRC Ispra European Commission'
        rootgrp.source = 'netCDF4 python module tutorial'
        latitudes.units = 'degrees_north'
        latitudes.point_spacing = 'even'
        longitudes.units = 'degrees_east'
        longitudes.point_spacing = 'even'
        u.units = 'm/s'
        v.units = 'm/s'
        p.units = 'hPa'
        times.units = 'hours since {}'.format(t0)


        p[:]=self.p
        times[:]=self.t
        latitudes[:]=self.lat
        longitudes[:]=self.lon
        u[:]=self.u
        v[:]=self.v
 
        rootgrp.close()

        
        
        
    def holland(self):
        
        cols=['w'+ name for name in radcols]
        
        varn=['rmse','dph','b','rmaxh','k','np.max(V)','vmax1']
        mcdic={el:[] for el in varn}
        
        hpar= pd.DataFrame(mcdic)
        
        # clean the values if present
        try:
           dcols=[unicode(x) for x in varn]
           self.data = self.data.drop(dcols,1)
        except Exception as e:
            print e
            pass
        
        for it in range(self.data.shape[0]):
   
            vmax=self.data.loc[it].vmax

        #
        #try:
        #    dp=tc.loc[it].dp
        #except:
        #    pass

            R=self.data.ix[it,radcols].values

            V=self.data.ix[it,cols].values

            vmax0vt = self.data.vmax0vt.values[it]

            time = self.data.time.values[it]

            sinfi = self.data.sinfi.values[it]

            lat=self.data.lat.values[it]

            w = R > 0.

            R = R[w].astype(float)

            V = V[w].astype(float)

            if R.size > 0 : 
                rmc = mc(R,V,sinfi,lat,vmax0vt)
            else:
                rmc = {el:np.nan for el in varn}

            df = pd.DataFrame(rmc,index=[it])

    
            hpar = hpar.append(df)
            
                
        self.data = pd.concat([self.data,hpar],axis=1)
        
    
    def output(self,filename):
        
        header=['time','xhc','yhc','b','k','rmax','deltap','vmax','vmax0','vtr','vtrx','vtry','bias','rmse']
        
        column_order=['time','lat','lon','b','k','rmaxh','dph','vmax','vmax0','vtr','vtrx','vtrxy','bias','rmse']
        
        outData.to_csv(filename,index=False, columns=column_order, sep='\t',header=header)
        
        
    def uvp(self,buffer=10,ni=100,nj=100,dt=5):
        
        tc = self.data.set_index('t')
        tc=tc.apply(pd.to_numeric) # convert to float
        
        tc=tc.dropna()
        
        #Define the big window
        minlon=tc.lon.min()-buffer
        maxlon=tc.lon.max()+buffer
        minlat=tc.lat.min()-buffer
        maxlat=tc.lat.max()+buffer  
        
        lons=np.linspace(minlon, maxlon,ni) # constract arrays
        lats=np.linspace(minlat, maxlat,nj)
        
        q1,q2=np.meshgrid(lons,lats) # create grid for the whole domain
        
        # we can now use pandas to interpolate every e.g. 5min
        tc_ = tc.resample('{}min'.format(dt)).mean()
        
        
        ux=[]
        uy=[]
        pp=[]
        for i in range(tc.shape[0]):
            bh,kh,dph,rmaxh,vtx,vty = tc.ix[i,['b','k','dp','rmax','vtrx','vtry']]
            zx,zy,pr=hvel2d(q1,q2,tc.lon[i],tc.lat[i],bh,kh,dph,rmaxh,vtx,vty)
            ux.append(zx)
            uy.append(zy)
            pp.append(pr)
            
        ux = np.array(ux)
        uy = np.array(uy)
        pp = np.array(pp)
        
                              
        self.u=ux
        self.v=ux
        self.p=pp
        self.lons=lons
        self.lats=lats
        
        
         
