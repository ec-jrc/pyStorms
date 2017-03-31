import numpy as np
import datetime


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
        self.shape = kwargs.get('shape', None)
        self.x     = kwargs.get('x', None)
        self.y     = kwargs.get('y', None)
    
    def parse(self,url):
    
    
    
    def writenc(self,filename):
        
        
        
    def output(self,filename):
        
        
        
