The shearing box approximation for modeling the local dynamics in accretion disks is implemented in AthenaK based on the methods described in [Stone & Gardiner 2010](https://ui.adsabs.harvard.edu/abs/2010ApJS..189..142S/abstract).

To enable simulations in the shearing box, set the boundary conditions in the x1-direction in the `<mesh>` block to `shear_periodic`, e.g.
```
    <mesh>
    nghost    = 3               # Number of ghost cells
    nx1       = 64              # Number of zones in X1-direction
    x1min     = -2.0            # minimum value of X1
    x1max     = 2.0             # maximum value of X1
    ix1_bc    = shear_periodic  # Inner-X1 boundary condition flag
    ox1_bc    = shear_periodic  # Outer-X1 boundary condition flag
```
In addition, add a new `shearing_box` block to the input file, e.g.
```
<shearing_box>
qshear = 1.5         # shear parameter (3/2 for Keplerian)
omega0 = 1.0         # orbital frequency
stratified = false   # true for stratified shearing box
```
The parameters in this block should be self-explanatory.

## Notes

- Orbital advection is implemented and is always used with the shearing box in AthenaK. It is not possible to select the orbital advection and shearing box algorithms independently.

- The shearing box and orbital advection algorithms work with both non-relativistic hydrodynamics and MHD, but not with relativistic hydro or MHD.

- The `/src/pgen/tests/shwave` problem generator along with input files in the `/inputs/shearing_box/` directory can be used to run shearing wave (shwave) tests.

- The `/src/pgen/mri2d` problem generator along with input files in the `/inputs/shearing_box/` directory can be used to run 2D MRI simulations in the $R-Z$ plane.  In 2D, only $R-Z$ coordinates have been implemented in the shearing box; 2D simulations in the $R-\phi$ plane are not implemented.

- The `/src/pgen/tests/mri3d` problem generator along with input files in the `/inputs/shearing_box/` directory can be used to run 3D MRI simulations, both for vertically stratified and unstratified disks.  Use the `<shearing_box>/stratified` parameter to select which type of problem to run.  For stratified shearing boxes, special vertical open boundaries are implemented in the `/src/pgen/tests/mri3d` problem generator, and must be enrolled by setting the boundary conditions in the x3-direction in the `<mesh>` block to `user`, e.g.
```
    <mesh>
    ix1_bc    = shear_periodic  # Inner-X1 boundary condition flag
    ox1_bc    = shear_periodic  # Outer-X1 boundary condition flag
    ix3_bc    = user            # Inner-X3 boundary condition flag
    ox3_bc    = user            # Outer-X3 boundary condition flag
```

- Be aware that with orbital advection, the azimuthal velocity $V_\phi$ (the x3-velocity in 2D $R-Z$ coordinates, or the x2-velocity in 3D) represents the velocity perturbations $\delta V_\phi$ from the background shear flow (e.g. Keplerian motion).  This must be taken into account when writing new problem generators, when analyzing the results written to output files, or if other terms depending on $V_\phi$ are included.

- For the reason above, viscosity is not yet implemented in the shearing box.

- Neither AMR nor SMR have been tested with the shearing box, and except for very special cases will almost certainly not work.