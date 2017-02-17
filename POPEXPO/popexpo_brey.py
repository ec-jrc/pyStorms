from pmap import *
import numpy as np
import pyresample


def get_pop(wlat,wlon,wind,nan):
    filename='../../POPEXPO/lspop20141.tif' #population data filename
    #define the lat/lon window of wind swath
    minlon=wlon[wind != nan].min()
    maxlon=wlon[wind != nan].max()
    minlat=wlat[wind != nan].min()
    maxlat=wlat[wind != nan].max()
    buf=[minlon,maxlon,minlat,maxlat]
    
    pop=getmap(filename,buf) # read population data
    
    mwind=np.ma.masked_array(wind,wind==nan) #mask the wind data for interpolation
    
    wgeo=pyresample.geometry.SwathDefinition(lons=wlon,lats=wlat) # the wind grid geometry
    pgeo=pyresample.geometry.SwathDefinition(lons=pop.lons,lats=pop.lats) # the pop grid geometry
    air_near=pyresample.kd_tree.resample_nearest(wgeo,np.flipud(mwind),pgeo,radius_of_influence=500000,fill_value=-999)
    
    reference = np.array([0,33,63,82,95,112,136,1000]) 
    reference = np.array([0,20,33,49,50,58,1000]) 

    output_labels = ['TD', 'TS' ,'CAT. 1','CAT. 2' , 'CAT. 3', 'CAT. 4', 'CAT. 5'] #The value assigned to each interval
    output_labels = ['', 'TS' ,'CAT. 1','CAT. 2' , 'CAT. 3', 'CAT. 4', 'CAT. 5'] #The value assigned to each interval

    sort_idx = np.argsort(reference) # saving the scale indeces 
    pos = np.searchsorted(reference[sort_idx], air_near, side = 'left') # sort the wind array
    out = np.argsort(reference)[pos] # restore the correct order of the scale, more info in http://stackoverflow.com/questions/31078160/classify-elements-of-a-numpy-array-using-a-second-array-as-reference
    
    dic={}
    for i in range(7):
        mask= out == i # wind force i
        try:
            npop = np.ma.masked_less(pop.data[mask],0).sum() # sum of people exposed to wind force i
            if npop is np.ma.masked :
                dic[output_labels[i]] = 0 
            else:
                dic[output_labels[i]] = npop
        except Exception as e :
            print e
    return dic


wind=getmap('/mnt/web/brey/u10max.map')

# compute lat/lon
widthw=wind.NCOLS
heightw=wind.NROWS
gw=wind.GeoTr
wminx = gw[0]                                                                                                                            
wminy = gw[3] + widthw*gw[4] + heightw*gw[5]                                                                                               
wmaxx = gw[0] + widthw*gw[1] + heightw*gw[2]                                                                                               
wmaxy = gw[3]                                                                                                                            
                                                                                                                                          
wlon=np.linspace(wminx,wmaxx,widthw,endpoint=True)                                                                                          
wlat=np.linspace(wminy,wmaxy,heightw,endpoint=True)                                                                                         


w=np.flipud(wind.data)  # the issue of geotiff orientation

mwind=np.ma.masked_array(w,w==wind.nan) # pay attention on the nan value

print mwind.max()

# full window expand
minlon=wlon.min()
maxlon=wlon.max()
minlat=wlat.min()
maxlat=wlat.max()


print minlon,maxlon,minlat,maxlat


xw,yw=np.meshgrid(wlon,wlat)


# ## compute pop exposure

buf=[minlon,maxlon,minlat,maxlat]

filename='./lspop20141.tif' #population data filename


pop=getmap(filename,buf) # read population data

mpop=np.ma.masked_array(pop.data,pop.data==pop.nan) # mask the nan values


wgeo=pyresample.geometry.SwathDefinition(lons=xw,lats=yw) # the wind grid geometry
pgeo=pyresample.geometry.SwathDefinition(lons=pop.lons,lats=pop.lats) # the pop grid geometry
air_near=pyresample.kd_tree.resample_nearest(wgeo,np.flipud(mwind),pgeo,radius_of_influence=500000,fill_value=wind.nan)


#reference = np.array([0,33,63,82,95,112,136,1000]) # in kts
reference = np.array([0,17,32,42,49,58,70,1000]) # in m/s

output_labels = ['TD', 'TS' ,'CAT. 1','CAT. 2' , 'CAT. 3', 'CAT. 4', 'CAT. 5'] #The value assigned to each interval
sort_idx = np.argsort(reference) # saving the scale indeces 
pos = np.searchsorted(reference[sort_idx], air_near, side = 'left') # sort the wind array
out = np.argsort(reference)[pos] # restore the correct order of the scale, more info in http://stackoverflow.com/questions/31078160/classify-elements-of-a-numpy-array-using-a-second-array-as-reference

dic={}
for i in range(7):
        mask= out == i # wind force i
        try:
            npop = np.ma.masked_less(pop.data[mask],0).sum() # sum of people exposed to wind force i
            if npop is np.ma.masked :
                dic[output_labels[i]] = 0 
            else:
                dic[output_labels[i]] = npop
        except Exception as e :
            print e

print dic


#exposure = get_pop(yw,xw,wind,wind.nan)


# In[37]:

#%%skip
resa=getmap('/mnt/web/brey/test.tif')


# In[38]:

resa


# In[39]:

# compute lat/lon
widthw=resa.NCOLS
heightw=resa.NROWS
gw=resa.GeoTr
wminx = gw[0]                                                                                                                            
wminy = gw[3] + widthw*gw[4] + heightw*gw[5]                                                                                               
wmaxx = gw[0] + widthw*gw[1] + heightw*gw[2]                                                                                               
wmaxy = gw[3]                                                                                                                            
                                                                                                                                          
wlon=np.linspace(wminx,wmaxx,widthw,endpoint=True)                                                                                          
wlat=np.linspace(wminy,wmaxy,heightw,endpoint=True)                                                                                         


# In[40]:

w=np.flipud(resa.data)  # the issue of geotiff orientation


# In[41]:

mwind=np.ma.masked_array(w,w==128) # pay attention on the nan value


# In[42]:

mwind.max()


# In[43]:

# full window expand
minlon=wlon.min()
maxlon=wlon.max()
minlat=wlat.min()
maxlat=wlat.max()


# In[44]:

xw,yw=np.meshgrid(wlon,wlat)


# In[45]:

parallels = np.arange(-90.,90,20.)                                                                                                        
meridians = np.arange(0.,360.,20.)                                                                                                        


# In[46]:

m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat, llcrnrlon=minlon,urcrnrlon=maxlon,resolution='l')    


# In[47]:

ti=np.arange(0.,mwind.max()+1) # custom range of colormap


# In[48]:

color=['w','b','g','y','r','purple']


# In[49]:

cmap = matplotlib.colors.ListedColormap(color[:mwind.max()+1]) # custom colormap


# In[50]:

m.contourf(xw,yw,mwind,cmap=cmap,animated=True)
m.colorbar(ticks=ti)
m.drawcoastlines(linewidth=1.5)                                                                                                           
m.drawparallels(parallels)                                                                                                                
m.drawmeridians(meridians)    


# In[51]:

#reference = np.array([0,33,63,82,95,112,136,1000]) # in kts


# In[59]:

reference = np.array([0,17,32,42,49,58,70,1000]) # in m/s


# In[60]:

output_labels = ['TD', 'TS' ,'CAT. 1','CAT. 2' , 'CAT. 3', 'CAT. 4', 'CAT. 5'] #The value assigned to each interval
sort_idx = np.argsort(reference) # saving the scale indeces 
pos = np.searchsorted(reference[sort_idx], mwind, side = 'left') # sort the wind array
out = np.argsort(reference)[pos] # restore the correct order of the scale, more info in http://stackoverflow.com/questions/31078160/classify-elements-of-a-numpy-array-using-a-second-array-as-reference


# In[53]:

popc=getmap('/mnt/web/brey/lspop20141_clipped.tif')


# In[54]:

popc.data.shape


# In[55]:

mwind.shape


# In[61]:

dic={}
for i in range(7):
        mask= out == i # wind force i
        try:
            npop = np.ma.masked_less(popc.data[mask],0).sum() # sum of people exposed to wind force i
            if npop is np.ma.masked :
                dic[output_labels[i]] = 0 
            else:
                dic[output_labels[i]] = npop
        except Exception as e :
            print e


# In[62]:

dic


# In[64]:

dic['TD']+dic['TS']


# In[ ]:



