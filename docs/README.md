TROPICAL CYCLONE ANALYSIS
==============================

This is a development project for using the Holland parametric model to produce wind and pressure profiles of Tropical Cyclones based on available data. 

## Getting Started

Tropical Cyclone (TC) attributes are estimated by corresponding operational agencies and centers around the world. For more info see the [INPUT.ipynb](./Notebooks/INPUT.ipynb) and [BestTrack.ipynb](./Notebooks/BestTrack.ipynb).

These data include location, windradii, pressure, max wind speed among others. However, there are missing info that preclude us from The proposed workflow consists of 4 steps : 

* Step 1 : From TC bulletins or Best Track create inpData file (see [INPUT.ipynb](./Notebooks/INPUT.ipynb), [Create inpData.ipynb](./Notebooks/Create inpData.ipynb), etc.).

* Step 2 : Compute translational and Coriolis velocities in order to move to a stationary frame (see [Subtract translational and Coriolis velocity.ipynb](./Notebooks/Subtract translational and Coriolis velocity.ipynb)).

* Step 3 : Estimate the parameters of the Holland Model and save outData file (see [Estimate Holland Parameters.ipynb](./Notebooks/Estimate Holland Parameters.ipynb)).

* Step 4 : Produce the wind and pressure profiles (see [Create Output.ipynb](./Notebooks/Output.ipynb)).


### Prerequisities

The required data can be freely downloaded by the corresponding listed sources. See Notebooks and tests for more details. 

A number of Python modules are required. A complete list is available in the file named requirements.txt.


## Tests

No tests are available at the moment.

## Authors

* **George Breyiannis** 


## Acknowledgments

* All the people that teach me stuff.  

## License
* The project is released under the EUPL 1.1 license. 

