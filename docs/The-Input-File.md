
### Input File Format
`AthenaK` requires an input file containing runtime parameters. Usually this file is given the name `<problem-name>.athinput`, where `<problem-name>` is an identifier of the problem. Sample input files are provided in `inputs/`.

Within the input file, parameters are grouped into named blocks, with the name of each block appearing on a single line within angle brackets, for example
```
    <time>
    cfl_number = 0.4       # The Courant, Friedrichs, & Lewy (CFL) Number
    nlim       = -1        # cycle limit
    tlim       = 1.0       # time limit
```
Block names must always appear in angle brackets on a separate line (blank lines above and below the block names are not required, but can be used for clarity).

Below each block name is a list of parameters, with syntax
```
    parameter = value [# comments]
```
White space after the parameter name, after the `=`, and before the `#` character is ignored. Everything after (and including) the `#` character is also ignored. Only one parameter value can appear per line. Comment lines (i.e. lines beginning with `#`) are allowed for documentation purposes. Both block names and parameter names are case sensitive. Note that tab should not be used in the input file as the parser does not support it.

### List of Input Blocks and Parameters
* `<comment>` (optional) : for documentation purposes
  * `problem` (optional) : problem description
  * `reference` (optional) : journal reference (if any)
  * `configure` (optional) : suggested configuration parameters
* `<job>` (**mandatory**)
  * `problem_id` (**mandatory**) : used in the output file names
* `<output[n]>` (optional) : output information (`[n]` is an integer)
  * `file_type` (**mandatory**) : file type (`vtk`, `bin`, `rst`, etc.; see Outputs)
  * `dt` (dt or dcycle is **mandatory**) : output interval in computing time
  * `dcycle` (dt or dcycle is **mandatory**) : output interval in cycles
  * `variable` (depends) : variable(s) to be output (see Outputs)
  * `data_format` (optional; used only for `.tab` and `.hst`) : format specifier string used for writing data (e.g. `%12.5e`)
  * `id` (optional) : output ID used in file names (default = `out[n]`)
  * `slice_x?` (optional) :  sliced output in orthogonal directions at the specified position (`? = 1, 2, or 3`)
  * `ghost_zones` (optional) : include boundary ghost cells in output (default = `false`)
* `<time>` (**mandatory**) : the CFL number and limit of the simulation
  * `cfl_number` (**mandatory**) : the Courant, Friedrichs, & Lewy (CFL) Number
  * `nlim` (**mandatory**) : time step limit (-1 = infinity)
  * `tlim` (**mandatory**) : time limit in computing time
  * `start_time` (optional) : time at the beginning of new simulation (default = 0)
  * `integrator` (optional) : time-integration scheme. Choices:
    * `rk2` (*default*) : second-order accurate Runge-Kutta/Heun's method
    * `rk3` : third-order accurate Strong Stability Preserving (SSP) variant
* `<mesh>` (**mandatory**) : grid configuration
  * `nx1, nx2, nx3` (**mandatory**) : the number of cells in the x1, x2, x3 directions, respectively (`nx3=1` means 2D, `nx2=1` implies 1D)
  * `x1min, x1max`, etc. (**mandatory**) : positions of the minimum and maximum surfaces (i.e., the box size)
  * `ix1_bc,ox1_bc`, etc. (**mandatory**) : boundary conditions (see Boundary Conditions)
  * `refinement` (optional) : enabling adaptive mesh refinement (default = none, see Adaptive Mesh Refinement)
  * `numlevel` (optional) : the number of AMR refinement levels (default = 1, see Adaptive Mesh Refinement)
  * `derefine_count` (optional) : the number of timesteps required before derefinement (default = 1, see Adaptive Mesh Refinement)
* `<meshblock>` (optional) : domain decomposition unit (see [[Using MPI and OpenMP]])
  * `nx1, nx2, nx3` (mandatory) : the number of the cells per MeshBlock (decomposition unit) in x1, x2, x3, respectively
* `<refinement[n]>` (optional) : (static) refinement regions (`[n]` is an integer, see Static Mesh Refinement)
  * `x1min, x1max,` etc. (mandatory) : positions of the refined regions
  * `level` (mandatory) : refinement level (root level = 0)

* `<hydro>` (**optional**) : parameters of hydrodynamics
  * `iso_sound_speed` (**mandatory** for isothermal EOS) : sounds speed in the isothermal EOS
  * `gamma` (**mandatory** for adiabatic EOS) : adiabatic index
  * `dfloor`, `pfloor` (optional) : density, pressure, and passive scalar concentration floors
  * `gamma_max` (optional) : maximum Lorentz factor in SR and GR

* `<mhd>` (**optional**) : parameters of MHD
  * `iso_sound_speed` (**mandatory** for isothermal EOS) : sounds speed in the isothermal EOS
  * `gamma` (**mandatory** for adiabatic EOS) : adiabatic index
  * `dfloor`, `pfloor` (optional) : density, pressure, and passive scalar concentration floors
  * `gamma_max` (optional) : maximum Lorentz factor in SR and GR (default = 1000)

* `<radiation>` (**optional**) : parameters of radiation
  * `nlevel`: (mandatory) refinement level for geodesic mesh
  * `rotate_geo`: (optional) rotate geodesic mesh to reduce grid alignment (default: true)
  * `angular_fluxes`: (optional) (default: true) flag to potentially disable angular fluxes
  * `reconstruct`: (optional) set radiation spatial reconstruction algorithm (options: `dc`, `ppm4`, `ppmx`, `wenoz`) (default: `plm`)
  * `rad_source`: (optional) enables radiation source term (default: true) 
  * `fixed_fluid`: (optional) if true, solve transport problem without evolving (magneto)hydrodynamics (default: false)
  * `affect_fluid`: (optional) if false, apply source term to intensity field, but do not feedback on fluid (i.e., disable source term update of $`T^0_\mu`$ from differences in pre- and post-coupling radiation moments; *this is non-conservative*) (default: true)
  * `arad`: (mandatory if `<units>` excluded and `<radiation>/rad_source=true`) radiation constant in code units
  * `kappa_s`: (mandatory if `<radiation>/rad_source=true`) scattering opacity in code units (or cgs units if `<units>` included)
  * `kappa_a`: (mandatory if `<radiation>/rad_source=true` and `<radiation>/power_opacity=false`) absorption opacity in code units (or cgs units if `<units>` included)
  * `power_opacity`: (optional) enable Kramer's type absorption opacity (default: false)
  * `beam_source`: (optional) flag to enable allocation of beam mask, for use with radiation beam source (default: false)
  * `dii_dt`: (mandatory if `<radiation>/beam_source`) dI/dt for angular bins flagged in beam mask

* `<coord>` (depends) : parameters for GR coordinate system
  * `a` (depends) : spin of black hole
  
* `<problem>` (optional) : problem-specific parameters

* `<chemistry>` (optional) : parameters of chemistry
  * `network`: (optional) The chemistry network to use. Options include: `H2`, and `GOW17`(upcoming). These networks often have their own runtime parameters which are detailed in the [Chemistry](Chemistry) page.
  * `ode_solver`: (optional) The ODE solver to use. See the [ODE Solver](ODE-Solver) page for which solvers are available and any runtime parameters they may have.

#### Changes from Athena++
In many instances, options that were specified at compile time in Athena++ are now specified at run time.  For example, the number of ghost zones, reconstruction algorithm, Riemann solver, ... TODO(@pdmullen).  Furthermore, the inclusion of certain blocks inside the input file determine which physics are incorporated into a given run... TODO(@pdmullen) 