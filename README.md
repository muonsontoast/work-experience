The plotting utility script has two main functions for generating plots. One generates only a top-down view of the arrangement of linear optics magnets in the Booster FODO lattice, and the other generates the same plot and an additional plot of the beta function values as a function of s around the lattice.

Quadrupole face and edge colours can be changed by overriding argument parameters.

The functions expect a n_FODOs argument (which should be an even number which is the total number of FODO cells in both of the curved sections of the Booster. It also expects a n_straight_FODOs argument as the number of FODO cells in a linear section of the lattice (where essential equipment like RF cavities are placed to supply energy to the beam).

The functions also expect drift and quadrupole length parameters as well as a FODO bend angle (in degrees) which should be twice the bend angle of a dipole since a FODO cell in the bent sections contains two such dipole elements to curve the trajectory of the particles between the quadrupoles.

For the beta plot, initial values for horizontal and vertical beta should also be supplied. If the wrong values are specified, beta beating can be observed where the FODO cell does not produce a complete closed oscillation.
