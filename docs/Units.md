`AthenaK` code units are dimensionless. The `<units>` infrastructure is designed to convert physical variables to an equivalent dimensionless quantity in code by a scaling factor, and vice versa. For any variable $`u`$, we denote its units as $`u_0`$, its value in code units as $`u_\mathrm{code}`$, and its value in physical units (e.g., cgs units) as $`u_\mathrm{phy}`$. Then we have

```math
    u_\mathrm{code}=u_\mathrm{phy}/u_0 \Longleftrightarrow u_\mathrm{phy}=u_\mathrm{code}u_0
```

For a unit system, the basic units include:
- length unit, $`l_0`$
- time unit, $`t_0`$
- mass unit, $`m_0`$

Then, other units can be derived from these three units, e.g., velocity unit $`v_0=l_0/t_0`$, density unit $`\rho_0=m_0/l_0^3`$, energy unit $`e_0=m_0v_0^2`$, and pressure unit $`P_0=e_0/l_0^3`$. As an option, mean molecular weight, $`\mu`$, can also be included in the units system to determine temperature unit. The temperature unit is given by $`T_0=P_0\mu m_\mathrm{H}/(\rho_0k_\mathrm{B})=v_0^2\mu m_\mathrm{H}/k_\mathrm{B}`$, where $`k_\mathrm{B}`$ is the Boltzmann constant and $`m_\mathrm{H}`$ is the atomic mass unit. In this way, we have $`P_\mathrm{code}=\rho_\mathrm{code}T_\mathrm{code}`$ in code.

The units system is initialized by the `<units>` block in the input file, including length, `length_cgs`, mass, `mass_cgs`, and time, `time_cgs`. As an option, mean molecular weight can also be initialized by `mu`. Default units are cgs units with $`\mu=1`$ if `<units>` block is added.


As an example, for simulations of the interstellar medium, a useful unit system is

```
<units>
length_cgs  = 3.0856775809623245e+18  # length is 1 pc
mass_cgs    = 6.195900228622575e+31   # number density is 1 cm^-3
time_cgs    = 3.15576e+13             # time is 1 Myr
mu          = 1.27                    # mean molecular weight
```

## General Relativity
In General Relativistic (Magneto)hydrodynamics, the system of equations are scale invariant.  In this way, there exists freedom in manipulating, in any one simulation, (1) the density scale $`\rho_0`$ to set the density, internal energy, and magnetic field strength in physical units and (2) the black hole mass $`M`$ to set the physical length scale $`l_0 = G M/c^2`$ (and hence timescale $`t_0 = l_0/c`$) where $`G`$ is the gravitational constant and $`c`$ is the speed of light (see [Wong et al. 2022](https://arxiv.org/abs/2202.11721) for a nice discussion).  

In General Relativistic Radiation (Magneto)hydrodynamics (see [Radiation](https://github.com/IAS-Astrophysics/athenak/wikis/Radiation)), however, the system of equations are no longer scale invariant.  Thus, one must specify the relevant physical scales in the problem.  Rather than specifying a length unit $`l_0`$, mass unit $`m_0`$, and time unit $`t_0`$, as described at the top, it is oft easier to specify a black hole mass $`M`$ and density unit $`\rho_0`$.  Given the gravitational constant $`G`$, speed of light $`c`$, and the black hole mass $`M`$, one can derive the length scale $`l_0`$ and time scale  $`t_0`$ in cgs units.  Combined with a density unit $`\rho_0`$, the mass scale  $`m_0`$ can then be derived.  Finally, given the mean molecular weight $`\mu`$ (same as above), one can infer the temperature unit $`T_0`$.  Therefore, when working with General Relativistic Radiation (Magneto)hydrodyanmics, we enable the user to specify

```
<units>
density_cgs = 1.0e-2  # density unit in g/cm3
bhmass_msun = 10.0    # black hole mass in solar masses
mu          = 0.5     # mean molecular weight (in amu)
```

**_Note_: In the above, note that the black hole mass **must be specified in solar masses**.  The conversion of the black hole mass to cgs units is handled by the `Units` API. 

In the codebase, one will notice that no `Units` member enters in the case of General Relativistic (Magneto)hydrodynamics (which makes sense given the scale invariance).  However, in the case of General Relativistic Radiation (Magneto)hydrodynamics, if the `<units>` block is specified, units are heavily used in the radiation source term (see the **Coupling** section of [Radiation](https://github.com/IAS-Astrophysics/athenak/wikis/Radiation)).   