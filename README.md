TROPICAL CYCLONE ANALYSIS
==============================

This is a development project for using the Holland parametric model to produce wind and pressure profiles of Tropical Cyclones based on available data. 

## Getting Started

Tropical Cyclone (TC) attributes are estimated by corresponding operational agencies and centers around the world. For more info see the [INPUT.ipynb](./INPUT.ipynb) and [BestTrack.ipynb](./BestTrack.ipynb).

These data include location, windradii, pressure, max wind speed among others. However, there are missing info that preclude us from The proposed workflow for estimating the wind field is the following 

* From TC bulletins or Best Track create inpData file (see [Create inpData.ipynb](./Create inpData.ipynb))

* Compute translational and Coriolis velocities in order to move to a stationary frame (see [Subtract translational and Coriolis velocity.ipynb](./Subtract translational and Coriolis velocity.ipynb))

* Estimate the parameters of the Holland Model and save outData file (see [Estimate Holland Parameters.ipynb](./Estimate Holland Parameters.ipynb))

* Produce the wind and pressure profiles (see [Create Output.ipynb](./Output.ipynb))


### Prerequisities

The required data can be freely downloaded by the corresponding listed sources. See Notebooks for more details. 

A number of Python modules are required. A complete list is available in a file named piplist.


## Tests

No tests are available at the moment.

## Authors

* **George Breyiannis** 


## Acknowledgments

* All the people that teach me stuff.  

## License
* The project is released under the GPL v3 license. 

  This library is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

