AthenaK supports a number of output options, each of which can be enabled by adding an `<output##>` block to the parameter file, e.g., `<output1>`, `<output2>`, etc. All output blocks must define the following parameters:
* `file_type`: the kind of output (e.g., full dumps, history files, restart files, etc.)
* `dt`: how frequently in code time to write this output

Some output blocks also support an `id` parameter which can be used to choose the stem for the filename. This is useful if the same variable needs to be output in different ways, e.g., having 2D sliced outputs in both the x-y and x-z planes at a relatively frequent cadence and full 3D grid dumps only occasionally.

## Full grid and sliced grid
AthenaK supports dumping grid variables to both legacy VTK (`file_type = vtk`) and custom AthenaK binary blobs (`file_type = bin`). **VTK support is limited to unigrid runs on a single MPI task and is therefore only useful for small tests**. For larger runs, binary files are more appropriate.

A full grid dump would be done as follows:
```
<output1>
file_type = bin # or vtk
variable  = mhd_w_bcc # write the MHD primitive variables with cell-centered B-fields
dt        = 100.0 # write out every 100.0 time units
id        = prim_3D # File stem; filename defaults to mhd_w_bcc.*.bin if not specified
```
`variable` specifies which quantity (or quantities) to output. A full list of quantities is available in `src/outputs/outputs.hpp`, but the following quantities are among the most commonly used:
* `hydro_u` or `mhd_u`: conserved hydrodynamical variables for specified fluid module, including any weighting by the volume form (i.e., they are densitized)
* `hydro_w` or `mhd_w`: primitive hydrodynamical variables

A slice through the $xy$ plane at $z=0$ could be dumped using:
```
<output2>
file_type = bin # or vtk
variable  = adm # write the ADM variables
dt        = 10.0 # write out every 10.0 time units
slice_x3  = 0.0 # slice along z=0
id        = adm_xy # File stem; filename defaults to adm.*.bin if not specified
```

A line in the $z$ direction at $x=0.5$, $y=0.25$ could be specified using:
```
<output3>
file_type = bin # or vtk or tab
variable  = hydro_u # write the conserved hydro variables
dt        = 1.0 # write out every 1.0 time units
slice_x1  = 0.5 # slice along x=0.5
slice_x2  = 0.25 # slice along y=0.25
```
Note that no interpolation is done during slicing operations, so the actual data written to the output file is sourced from the nearest cell center to the left of the desired position, e.g., if there are cell centers at $x=0.3$ and $x=0.55$, slicing through $x=0.5$ will return the cell at $x=0.3$. This means that meshblocks sourced from different refinement levels will be output at slightly different points.

Data can also be written to ASCII tables by setting `file_type = tab`, but only in one dimension. Therefore, multidimensional grids must be sliced before outputting to tabulated data. By default, `tab` outputs are written with six significant figures. This can be changed by setting `data_format` to the corresponding C-string format specifier (see the discussion below on history outputs).

## Restart files
Restart (or checkpoint) outputs dump the entire state of the run at a regular interval so that the run can be restarted from that point later on.
```
<output4>
file_type = rst
dt        = 1000.0 # dump a checkpoint every 1000.0 time units
```
Restarting is as simple as passing the `-r` flag with the restart file to AthenaK:
```bash
./athena -r rst/my_run.00001.rst
```
All input parameters are stored inside the restart file, so it is not necessary to pass an input file when restarting. However, a parameter file passed to AthenaK during a restart will take precedence over the parameters in the restart file:
```bash
./athena -r rst/my_run.00001.rst -i new_params.athinput
```
Note that while it is possible to update parameters in existing physics modules this way, adding or subtracting physics during restarts via a new parameter file is not directly supported at this time.

## History outputs
History outputs are global reductions of the entire grid at a given time. For example, hydro and MHD simulations will compute global integrals of the conserved variables to measure things like the total mass or momentum in the system, and Z4c simulations will calculate the L2 norm of various constraint monitors. It is also possible to add custom history outputs via the problem generator, such as integrating various fluxes across spherical surfaces in `gr_torus` or tracking the maximum density as in `dyngr_tov`.

Below is an example of a history file output:
```
<output5>
file_type = hst  # history output
dt        = 10.0 # output interval
```
Additionally, if the pgen supports a custom history function, the input file should set `problem/user_hist = true` and may optionally set `output#/user_hist_only = true` if only the user history file is desired:
```
<problem>
... # other parameters
user_hist = true # enroll user-defined history

...
<output1>
file_type = hst       # history output
dt        = 10.0      # output interval
user_hist_only = false # set to true to output only the user-defined history function
```

History files are human-readable text files with columns separated by whitespace and comments marked with `#` as in Python and other scripting languages. Consequently, these files can be readily imported into numpy, gnuplot, etc. By default, outputs are written with six digits. This can be modified by setting the `data_format` option. For example,
```
<output2>
file_type = hst       # history output
dt        = 10.0      # output interval
data_format = %20.15e # write out 15 significant digits
```
will write 15 significant digits in scientific notation, using whitespace to ensure the string width is at least 20 characters. Formatting strings use C-style formatting specifiers, as in [`printf`](https://cplusplus.com/reference/cstdio/printf/).

## PDF/histograms
TODO

## Interpolated spherical grid
AthenaK can interpolate data to a latitude/longitude spherical grid and save it to a `.vtk` file. An example of this is shown below:
```
<output3>
file_type = sph       # spherical grid output
dt        = 5.0       # output interval
variable  = mhd_w_bcc # write out MHD variables
radius    = 200.0     # extract data at a radius of 200 in code units
ntheta    = 64        # number of angles in polar direction, 32 by default; azimuthal direction is always 2*ntheta.
xc        = 0.0       # x, y, and z coordinates of center of sphere; (0, 0, 0) by default.
yc        = 0.0
zc        = 0.0
```

## Interpolated cartesian grid
TODO

## Coarsened grids
TODO