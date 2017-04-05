# Compute distance in km between 2 locations in lat/lon
import numpy as np
from math import radians, cos, sin, asin, sqrt
import xml.etree.ElementTree as et 
from xml.dom import minidom
from parameters import *


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = et.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def pres(r,bh,dph,rmaxh):
    return pn+dph*np.exp(-(rmaxh/r)**bh)
    
def hvel(r,dph,bh,rmaxh,kh):
        ratio=(r-rmaxh)/(maxR-rmaxh)
        ratio2=np.minimum(np.maximum(ratio,0),1)# control the slope at high radius
        x=0.5+ratio2*kh
        return (bh/rhoa*(rmaxh/r)**bh*dph*np.exp(-(rmaxh/r)**bh))**x

def hvel2d(l1,l2,lon0,lat0,bh,kh,dph,rmaxh,vtx,vty):
    r = Haversine(l1, l2, lon0, lat0)
    ratio=(r-rmaxh)/(maxR-rmaxh)
    ratio2=np.minimum(np.maximum(ratio,0),1)# control the slope at high radius
    xh=0.5+ratio2*kh
    theta=np.arctan2((l2-lat0),(l1-lon0))
    fcor = 2*omega*np.sin(theta) #coriolis force
    ur=(bh/rhoa*(rmaxh/r)**bh*dph*np.exp(-(rmaxh/r)**bh)+(r*fcor/2.)**2)**xh-r*fcor/2.
    ux=-ur*np.sin(theta)
    uy=ur*np.cos(theta)
    return ux+vtx*(1.-np.exp(-(rmaxh/r)**bh)),uy+vty*(1.-np.exp(-(rmaxh/r)**bh)), pres(r,bh,dph,rmaxh)


def Haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1 = np.radians([lon1, lat1])
    lon2, lat2 = np.radians([ lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = 6378.388 * c
    m = km * 1000
    return  m