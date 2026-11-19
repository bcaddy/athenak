**If you use this chemistry module in your publication, please cite the accompanying code paper [Gong et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...42G/abstract).**
The paper also provides some more details beyond this documentation.

# Formulation

![chemistry flowchart](https://i.imgur.com/JnpD8qM.png)
Chemistry uses the basic functionality of the passive scalars included in the `hydro` or `mhd` modules. The reaction rates act as a source term for the passive scalars. It can also interact with other parts of the code, e.g. heating and cooling rates in the energy equation, the mean molecular weight, opacity in radiation transfer, etc.

# Solving Chemistry

Because the chemical reaction rates for different species often differ by orders of magnitude, implicit solvers are better suited to solve the rate equations. We use the operator split method: after the `hydro`/`mhd` solver has finished, we advance the chemical abundances and the internal energy for the hydrodynamic time step dt, by solving a set of coupled ODEs of chemical rate equations and heating/cooling implicitly:

$$\frac{\mathrm{d} x_s}{\mathrm{d} t} = \mathcal{C}_s$$

$$\frac{\mathrm{d} e}{\mathrm{d} t} = n(\Gamma - \Lambda)$$

In the code, the reaction rate $\mathcal{C}_s$ and the $\frac{de}{dt}$ (heating and cooling) are computed in the `evaluate_function` method of the various chemistry network classes. Because the chemical reaction rate $\mathcal{C}_s$ usually depends on the temperature `T`, the user often need to have some expression of `T` as a function of the internal energy `e` in their RHS. Currently, chemistry works with a adiabatic equation of state with a fixed adiabatic index set by the `gamma` parameter in the `hydro` or `mhd` modules. General EOS dependence on chemistry has not been included yet.

Note that the passive scalar advection (and diffusion) is solved separately prior to the chemical rate equations. This operator split method is only first-order accurate. The forced conservation of elemental abundances (such as by rescaling the flux of different species) has not been implemented yet. Relatively small errors are usually found for elemental conservation from ODE solvers.

# Running the Code

## Input File

To run with chemistry you need to include the `<chemistry>` block in the input file. Within that you have the following parameters

| Option                 | Type   | Allowed Values            | Default              | Description                                                                                                                                 |
| ---------------------- | ------ | ------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| network                | string | H2, GOW17, NCR            | None, mandatory flag | The chemistry network to use. These networks often have their own runtime parameters which are detailed in their respective sections below. |
| ode_solver             | string | forward_euler, kokkos_BDF | None, mandatory flag | The ODE solver to use. See the [ODE Solvers](ODE-Solvers) page for which solvers are available and any runtime parameters they may have.    |
| mu_H                   | Real   | any real > 1              | 1.4                  | The mean molecular mass per hydrogen nucleon                                                                                                |

Chemistry also requires that the `<units>` module be enabled. A good default set of units to use for ISM chemistry is:

```
<units> 
# ISM units
length_cgs  = 3.0856775809623245e+18  # length is 1 pc
mass_cgs    = 6.195900228622575e+31   # number density is 1 cm^-3
time_cgs    = 3.15576e+13             # time is 1 Myr
mu          = 1.4                     # mean molecular weight
```

# Chemistry

## Passive Scalars

The chemical species uses the passive scalar structure in the code. By default, the number of species is the number of scalars. If the user wants additional passive scalars to be added (for example, tracer scalars that is not involved in chemical reactions), the number of scalars can be set separately in the `nscalars` parameter.

## Existing Chemical Networks

### H2

This is a 2-species HI-H2 network. Although this network is very simplified and does not represent the realistic HI-H2 transition in the ISM, it has the advantage of having an analytic solution when $C_v$ is constant, and is used in tests.

Runtime parameters for the H2 network, these go inside the `<chemistry>` block:

| Option                 | Type   | Allowed Values | Default       | Description                                                                                    |
| ---------------------- | ------ | -------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| h2_constant_cv         | bool   | true, false    | false         | Whether or not $C_v$ should be constant. If true then $C_v$ is set to $1.1 K_b / (\gamma - 1)$ |
| h2_isothermal          | bool   | true, false    | false         | If the H2 network should use an isothermal EOS, i.e. set the change in internal energy to zero |

### GOW17

The GOW17 network is from the work of [Gong, Ostriker and Wolfire (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...843...38G/abstract). It is a widely used and well-tested network for carbon and oxygen chemistry in the atomic and molecular ISM. It has the advantage of being relatively simple, with only 18 species and about 50 reactions, while still accurately capturing the most important chemical and thermal processes compared to more sophisticated networks.

Of the 18 species, 12 are evolved directly by the ODE solver (He+, OHx, CHx, CO, C+, HCO+, H2, H+, H3+, H2+, O+, and Si+) while the remaining "ghost" species (Si, C, O, He, e-, and H) are recomputed algebraically from the conservation laws on every function evaluation. The internal energy is evolved alongside the species for a total of 13 equations. Because the reaction rates span many orders of magnitude the network is stiff, so an implicit ODE solver such as [Kokkos BDF](ODE-Solvers#kokkos-bdf) is recommended and the Forward Euler solver is known not to converge in a reasonable amount of time.

The GOW17 network uses 8 radiation bands: 6 photodissociation and photoionization bands, 1 band for photoelectric heating and grain-assisted recombination, and a final band that holds the cosmic-ray ionization rate. See [Radiation](#radiation) for how these are set.

Runtime parameters for the GOW17 network, these go inside the `<chemistry>` block:

| Option                           | Type | Allowed Values         | Default                 | Description                                                                                                                    |
| -------------------------------- | ---- | ---------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------|
| GOW17_Z_d                        | Real | any real $\geq$ 0      | 1.0                     | Dust metallicity relative to solar                                                                                             |
| GOW17_Z_g                        | Real | any real $\geq$ 0      | 1.0                     | Gas-phase metallicity relative to solar. Scales the `GOW17_xC`, `GOW17_xO`, and `GOW17_xSi` abundances                         |
| GOW17_xHe                        | Real | any real $\geq$ 0      | 0.1                     | Helium abundance per hydrogen nucleon                                                                                          |
| GOW17_xC                         | Real | any real $\geq$ 0      | 1.6e-4                  | Carbon abundance per hydrogen nucleon at $Z=1$ (multiplied by `GOW17_Z_g`)                                                     |
| GOW17_xO                         | Real | any real $\geq$ 0      | 3.2e-4                  | Oxygen abundance per hydrogen nucleon at $Z=1$ (multiplied by `GOW17_Z_g`)                                                     |
| GOW17_xSi                        | Real | any real $\geq$ 0      | 1.7e-6                  | Silicon abundance per hydrogen nucleon at $Z=1$ (multiplied by `GOW17_Z_g`)                                                    |
| GOW17_isothermal                 | bool | true, false            | false                   | Turn off heating and cooling and use a fixed temperature, $T = c_s^2 \mu_\mathrm{iso} m_\mathrm{H} / k_B$, for the reaction rates. The hydro EOS is unchanged, so with an ideal EOS the internal energy is still evolved by the hydro. If true, `GOW17_mu_iso` and `hydro/iso_sound_speed` must also be set |
| GOW17_mu_iso                     | Real | any real > 0           | mandatory if isothermal | Mean molecular weight used to convert `hydro/iso_sound_speed` into the fixed temperature                                       |
| GOW17_temperature_min_rates      | Real | any real > 0           | 1.0                     | Minimum temperature (K) for the reaction rates; also applied as a temperature floor in the energy equation                     |
| GOW17_temperature_max_rates      | Real | any real > 0           | $\infty$                | Maximum temperature (K) for the reaction rates. Does not apply to collisional dissociation reactions or the energy equation    |
| GOW17_temperature_max_heating    | Real | any real > 0           | $\infty$                | Temperature (K) above which heating is turned off                                                                              |
| GOW17_temperature_min_cooling    | Real | any real > 0           | 1.0                     | Temperature (K) below which cooling is turned off                                                                              |
| GOW17_temperature_max_cooling_nm | Real | any real > 0           | 1.0e9                   | Temperature (K) at which cooling for the neutral medium is capped                                                              |
| GOW17_Leff_CO_max                | Real | any real > 0           | 3.0e20                  | Maximum effective length (cm) for CO cooling                                                                                   |
| GOW17_H2_rovib_cooling           | bool | true, false            | true                    | Whether to include H2 rovibrational cooling                                                                                    |
| is_kgrH2_const                   | bool | true, false            | false                   | Whether to use a constant H2 formation rate on grains                                                                          |

### NCR

The NCR network is the reduced hydrogen, carbon, and oxygen photochemistry network of TIGRESS-NCR, [Kim et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..264...10K/abstract). It is cheap enough for large galactic-disk simulations while following the cooling and heating of the cold, warm, and hot ISM. This is a port of the TIGRESS-NCR implementation, see [Differences from TIGRESS NCR](#differences-from-tigress-ncr).

Only H2 and H+ are evolved by the ODE solver, along with the internal energy, for a total of 3 equations. Everything else is algebraic and recomputed on every function evaluation:

- $x_\mathrm{HI} = 1 - x_\mathrm{H^+} - 2 x_\mathrm{H_2}$, from conservation.
- The electron abundance $x_e$ is the solution of $x_e = F(x_e, T)$ at the temperature set by the internal energy, which itself depends on $x_e$. Below `NCR_temp_hot0`, $F = x_\mathrm{H^+} + x_\mathrm{C^+} + x_\mathrm{O^+} + x_{e,\mathrm{mol}}$, with $x_\mathrm{C^+}$ in equilibrium, $x_\mathrm{O^+}$ tied to $x_\mathrm{H^+}$ by charge exchange, and $x_{e,\mathrm{mol}}$ from molecular ions. Above it, $F = x_\mathrm{H^+} + x_{e,\mathrm{He}}(T) + Z_g\, x_{e,\mathrm{metal}}(T)$ from the collisional ionization equilibrium (CIE) table of TIGRESS-NCR, which is embedded in the code. The pair is solved together with a bracketed secant method. $x_e$ is not stored.
- CO from the equilibrium fit of TIGRESS-NCR, and neutral C and O from conservation.
- He is fixed at $x_\mathrm{He} = 0.1$.

Heating is by cosmic rays, the photoelectric effect, and H2 formation, pumping, and dissociation. Cooling is by H2 rovibrational lines ([Moseley et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.3290M/abstract)), C+, CI, OI, CO, Lyα, HI collisional ionization, H recombination and free-free emission, recombination on grains, and nebular lines, and above `NCR_temp_hot0` by the CIE cooling of He and metals, which is blended in with a sigmoid. Above `NCR_temp_hot1` H2 cooling and heating and the cold-gas terms are off. The CO cooling uses the constant velocity gradient `NCR_dvdr`, as TIGRESS-NCR does. The network is stiff, so [Kokkos BDF](ODE-Solvers#kokkos-bdf) is recommended.

The rate coefficients can be computed exactly or interpolated from tables built at startup, set by `NCR_interp_mode`, as in TIGRESS-NCR: 0 is exact, 1 interpolates in log-log, and 2 and 3 use the NQT2 and NQT1 approximations of the logarithm. The tables hold the case B H recombination and HI collisional ionization rates up to high temperature, and the C+ recombination, H2 grain formation, and C+, CI, and OI collisional de-excitation rates in cold gas.

Runtime parameters for the NCR network, these go inside the `<chemistry>` block:

| Option            | Type | Allowed Values       | Default | Description                                                                                                                                         |
| ----------------- | ---- | -------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| NCR_Z_g           | Real | any real $\geq$ 0    | 1.0     | Gas-phase metallicity relative to solar. Scales `NCR_xC`, `NCR_xO`, and the CIE metal electrons and cooling                                           |
| NCR_Z_d           | Real | any real $\geq$ 0    | 1.0     | Dust metallicity relative to solar. Scales grain processes and, with six-ray radiation, the dust extinction                                         |
| NCR_xC            | Real | any real $\geq$ 0    | 1.6e-4  | Carbon abundance per hydrogen nucleon at $Z=1$ (multiplied by `NCR_Z_g`)                                                                             |
| NCR_xO            | Real | any real $\geq$ 0    | 3.2e-4  | Oxygen abundance per hydrogen nucleon at $Z=1$ (multiplied by `NCR_Z_g`)                                                                             |
| NCR_temp_hot0     | Real | $\geq 10^4$          | 2e4     | Temperature (K) above which $x_e$ comes from the CIE table and the CIE cooling is blended in. It must be at least $10^4$ K, the bottom of the table  |
| NCR_temp_hot1     | Real | > `NCR_temp_hot0`    | 3.5e4   | Temperature (K) above which H2 formation, H2 cooling and heating, and the cold-gas cooling are off                                                 |
| NCR_temp_mu_floor | Real | any real > 0         | 2.0     | $T/\mu$ (K) at or below which net cooling is switched off; net heating still applies                                                                |
| NCR_dvdr          | Real | any real $\geq$ 0    | 3.240779289444365e-14 | Velocity gradient ($\mathrm{s^{-1}}$) of the CO LVG cooling, a constant input as TIGRESS's `dvdr`. The default is 1 km/s/pc. The effective gradient is at least $v_\mathrm{th,CO}/100$ pc |
| NCR_interp_mode   | int  | 0, 1, 2, 3           | 1       | How the rate coefficients are computed: 0 exact, 1 log-log tables, 2 NQT2 tables, 3 NQT1 tables                                                     |
| NCR_chem_only     | bool | true, false          | false   | Hold the temperature fixed, at its value at the start of each step, while the abundances evolve. The internal energy is adjusted as $x_e$ changes. Used by the equilibrium test |

The NCR network uses 5 radiation bands. They are chosen so that both [constant](#constant-radiation) and [six-ray](#six-ray) radiation give NCR's fields without NCR-specific radiation code:

| Band       | Field                                                                                              | Six-ray attenuation                                                                         |
| ---------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `iph_LW`   | $\chi_\mathrm{LW}$, the Lyman-Werner field. Also used for CO photodissociation                     | dust, $\gamma = 3.74$, i.e. $\sigma_{d,\mathrm{LW}} = 2\times10^{-21}\ \mathrm{cm^2}$       |
| `iph_H2`   | $\chi_\mathrm{H_2}$, the field that dissociates H2                                                 | dust ($\gamma = 3.74$) and H2 self-shielding                                                |
| `iph_C`    | $\chi_\mathrm{CI}$, the field that ionizes neutral carbon                                          | dust ($\gamma = 3.74$) and shielding by H2 (no C self-shielding, see below)                 |
| `irad_GPE` | $\chi_\mathrm{PE}$, the photoelectric heating field                                                | dust, $\sigma_\mathrm{PE} = 10^{-21}\ \mathrm{cm^2}$                                        |
| `irad_CR`  | $\xi_\mathrm{CR}$, the cosmic-ray ionization rate per H                                            | as GOW17, with `radiation_CR_shielding`                                                     |

The FUV field is the ISRF-weighted mean of $\chi_\mathrm{PE}$ and $\chi_\mathrm{LW}$. With six-ray radiation the dust extinction uses `NCR_Z_d`. NCR has no neutral carbon in its column densities, so the C band gets H2 shielding only. There are no ionizing (LyC) photons, $\xi_\mathrm{ph,HI} = \xi_\mathrm{ph,H_2} = 0$.

#### Differences from TIGRESS NCR

The equilibria match TIGRESS-NCR, see the [NCR Equilibrium](#ncr-equilibrium) test. The known differences are:

- $x_e$ is solved with the temperature in every function evaluation rather than lagged by a step. Transients differ, equilibria do not.
- H2 above `NCR_temp_hot1` decays by collisional dissociation instead of being set to zero.
- Constant radiation gives $\chi_\mathrm{H_2} = \chi_\mathrm{LW}$. TIGRESS applies the H2 self-shielding factor at zero column, 0.99994, a $6\times10^{-5}$ effect.
- Six-ray radiation is new for NCR: it uses the H2 self-shielding and carbon shielding functions shared with GOW17, with no neutral carbon column, and dust-only shielding of CO.
- TIGRESS uses compiled-in values for some thresholds and abundances and input values for others. AthenaK uses the inputs everywhere: `NCR_temp_hot0` and `NCR_temp_hot1`, and `NCR_xC` and `NCR_xO`.
- Not ported: the adaptive ray tracing (ART), LyC, and $\xi_\mathrm{ph}$ paths; the local shielding length and local cosmic-ray attenuation; dust temperature balance; TIGRESS's option switches, of which only the defaults are implemented; TIGRESS's explicit subcycling solver; and the feedback of the cooling time on the hydro timestep.
- Small-value floors and divide-by-zero guards are $10^{-20}$, as in the rest of AthenaK, instead of TIGRESS's $10^{-50}$. Floors applied before taking a logarithm of the rate and CIE tables are $1000$ times the smallest normalized floating-point number, as TIGRESS.
- With $\xi_\mathrm{CR} = 0$ the CO abundance uses the limit of the CO critical density, where TIGRESS gives NaN.

## Radiation

Chemistry can depend on the radiation field, for example through photodissociation, photoionization, and photoelectric heating. The radiation method is selected at runtime with the `radiation_integrator` parameter in the `<chemistry>` block. Other radiation modules in AthenaK, such as the `<radiation>` transport module, are not coupled to chemistry at this time.

| Option               | Type   | Allowed Values    | Default  | Description                             |
| -------------------- | ------ | ----------------- | -------- | --------------------------------------- |
| radiation_integrator | string | constant, six_ray | constant | The radiation method used for chemistry |

The radiation field is stored as one value per frequency band. The number and meaning of the bands depend on the chemistry network; see the network sections above.

### Constant Radiation

The radiation field is uniform in space and constant in time. This is the simplest option. It is useful for testing and as a template for writing your own radiation method.

Runtime parameters for constant radiation, these go inside the `<chemistry>` block:

| Option       | Type | Allowed Values    | Default | Description                                                                                                    |
| ------------ | ---- | ----------------- | ------- | -------------------------------------------------------------------------------------------------------------- |
| radiation_G0 | Real | any real $\geq$ 0 | 0       | Strength of the interstellar radiation field relative to the Draine (1978) field, applied to every photon band |
| radiation_CR | Real | any real $\geq$ 0 | 2e-16   | Cosmic-ray ionization rate per hydrogen nucleon in s$^{-1}$                                                    |

### Six-ray

The six-ray method computes the radiation field by ray tracing along the six Cartesian directions ($\pm x$, $\pm y$, $\pm z$) and averaging the attenuated field over those directions. The incident radiation is assumed to come from the edge of the computational domain along each ray, and it can be set separately for each direction. Because the incident field is assumed to come from the physical boundary, it is compatible with periodic, outflow, reflecting, and user-defined boundary conditions.

#### Column densities

For every cell, the column densities of H nuclei ($N_\mathrm{H}$), H2 ($N_\mathrm{H_2}$), CO ($N_\mathrm{CO}$), and neutral carbon ($N_\mathrm{C}$) are summed along each of the six rays, from the domain boundary the ray enters through up to the cell. The cell's own column is included with weight `shielding_fraction_cell`, 0.5 by default, so each cell is shielded by half of itself. The neutral carbon abundance comes from carbon conservation, $x_\mathrm{C} = x_\mathrm{C,tot} - x_\mathrm{HCO^+} - x_\mathrm{CH_x} - x_\mathrm{CO} - x_\mathrm{C^+}$, floored at zero. Column densities are converted to cm$^{-2}$ with the `<units>` module and `mu_H`.

The six sums are computed concurrently with the `LineScan` utility, which handles MPI decomposition and static and adaptive mesh refinement. The columns are recomputed once per timestep at the beginning of the chemistry stage.

#### Shielded radiation field

Each ray $i$ starts with the incident field $G_{0,i}$ of the boundary it enters through. Along the ray:

- The dust extinction is $A_V = N_\mathrm{H} Z_d / 1.87\times10^{21}\ \mathrm{cm^{-2}}$, where $Z_d$ is `GOW17_Z_d`.
- Each photodissociation and photoionization band $f$ is attenuated by dust, $G_{0,i}\, e^{-\gamma_f A_V}$, with $\gamma_f$ = 3.76 (C), 2.12 (CHx), 3.88 (CO), 2.66 (OHx), 4.18 (H2), and 2.61 (Si).
- The H2, CO, and C bands are additionally multiplied by self-shielding factors:
  - H2 self-shielding, from [Draine & Bertoldi (1996)](https://ui.adsabs.harvard.edu/abs/1996ApJ...468..269D/abstract) equation (37), with a Doppler parameter of 3 km/s.
  - CO self-shielding and shielding by H2, from the tables of [Visser, van Dishoeck & Black (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...503..323V/abstract).
  - Neutral carbon self-shielding and shielding by H2, from [Tielens & Hollenbach (1985)](https://ui.adsabs.harvard.edu/abs/1985ApJ...291..722T/abstract).
- The photoelectric heating band is attenuated as $G_{0,i}\, e^{-N_\mathrm{H} \sigma_\mathrm{PE} Z_d}$ with $\sigma_\mathrm{PE} = 10^{-21}\ \mathrm{cm^2}$.
- The cosmic-ray band is `radiation_CR`. If `radiation_CR_shielding` is true, it is reduced to `radiation_CR` $\times\, N_0 / N_\mathrm{H}$ where $N_\mathrm{H} > N_0 = 9.35\times10^{20}\ \mathrm{cm^{-2}}$.

The field the network sees in each band is the average over all six rays, i.e. the sum divided by six. Rays with no incident field still count in the average, so a slab lit from a single face sees at most $G_0/6$.

Runtime parameters for six-ray radiation, these go inside the `<chemistry>` block:

| Option                  | Type | Allowed Values         | Default        | Description                                                                                                                       |
| ----------------------- | ---- | ---------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| radiation_G0            | Real | any real $\geq$ 0      | 0              | Incident radiation field relative to the Draine (1978) field, the default for every face                                         |
| radiation_G0_inner_x1   | Real | any real $\geq$ 0      | `radiation_G0` | Incident field of the ray entering through the inner x1 boundary, travelling in $+x$                                              |
| radiation_G0_outer_x1   | Real | any real $\geq$ 0      | `radiation_G0` | Incident field of the ray entering through the outer x1 boundary, travelling in $-x$                                              |
| radiation_G0_inner_x2   | Real | any real $\geq$ 0      | `radiation_G0` | Incident field of the ray entering through the inner x2 boundary, travelling in $+y$                                              |
| radiation_G0_outer_x2   | Real | any real $\geq$ 0      | `radiation_G0` | Incident field of the ray entering through the outer x2 boundary, travelling in $-y$                                              |
| radiation_G0_inner_x3   | Real | any real $\geq$ 0      | `radiation_G0` | Incident field of the ray entering through the inner x3 boundary, travelling in $+z$                                              |
| radiation_G0_outer_x3   | Real | any real $\geq$ 0      | `radiation_G0` | Incident field of the ray entering through the outer x3 boundary, travelling in $-z$                                              |
| radiation_CR            | Real | any real $\geq$ 0      | 2e-16          | Unshielded cosmic-ray ionization rate per hydrogen nucleon in s$^{-1}$                                                            |
| radiation_CR_shielding  | bool | true, false            | false          | Whether to attenuate the cosmic-ray ionization rate above $N_\mathrm{H} = 9.35\times10^{20}\ \mathrm{cm^{-2}}$                  |
| shielding_fraction_cell | Real | 0 to 1                 | 0.5            | Fraction of a cell's own column that shields the cell. 0 excludes the cell itself and 1 includes all of it                         |

## Timestep details

Currently, the overall chemical timestep is the same as the hydro timestep: the ODE solver advances the chemical abundances and energy for a single hydro timestep. Note that the ODE solver uses subcycling internally, and the subcycling timestep is adjusted to ensure the solution converges, but this process is hidden from the user. The overall timestep can be reduced by lowering the CFL number or by setting a maximum timestep with `dt_max` in the `<time>` block, and the ODE solvers generally have runtime options to make them more accurate. `dt_max` is useful when the hydro places little or no limit on the timestep, e.g. kinematic evolution with zero velocity. The six-ray column densities are recomputed once per timestep, so in that case the number of timesteps sets how many times the radiation and chemistry are brought into balance, as in the static PDR test below.

## Test Problems

### H2 Uniform

This test problem has a constant hydrodynamic background and follows the evolution of the H and H2 abundances. It has an analytic solution and details can be found in Section 4.1 of [Gong et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...42G/abstract).

- Input File: [H2_uniform.athinput](../../blob/main-chemistry/inputs/chemistry/H2_uniform.athinput)
- `pgen_name`: H2_uniform

Problem generator parameters, all are within the `<problem>` block unless otherwise specified:

| Option                 | Type   | Default       | Description                                                                        |
| ---------------------- | ------ | ------------- | ---------------------------------------------------------------------------------- |
| n_H                    | Real   | mandatory     | The number density of hydrogen                                                     |
| hydro/iso_sound_speed  | Real   | mandatory     | The isothermal sound speed defined in the `<hydro>` block                          |
| vx_kms                 | Real   | mandatory     | The x velocity in km/s                                                             |
| init_H                 | Real   | mandatory     | The initial abundance of hydrogen gas                                              |
| init_H2                | Real   | mandatory     | The initial abundance of molecular hydrogen                                        |

#### Time Evolution of the H2 Uniform Test

![H2 Uniform Temperature](uploads/chemistry/H2_uniform_Temperature.svg)
![H2 Uniform Specific Internal Energy](uploads/chemistry/H2_uniform_Specific_Internal_Energy.svg)
![H2 Uniform H2 Abundance](uploads/chemistry/H2_uniform_H2_Abundance.svg)
![H2 Uniform H Abundance](uploads/chemistry/H2_uniform_H_Abundance.svg)

### H2 Advecting Gaussian

This test problem has a Gaussian profile in H and H2 abundances with a constant velocity in $x$. It has an analytic solution and details can be found in Section 4.1 of [Gong et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...42G/abstract).

- Input File: [H2_advection.athinput](../../blob/main-chemistry/inputs/chemistry/H2_advection.athinput)
- `pgen_name`: H2_advection

Problem generator parameters, all are within the `<problem>` block unless otherwise specified:

| Option                 | Type   | Default       | Description                                                                        |
| ---------------------- | ------ | ------------- | ---------------------------------------------------------------------------------- |
| n_H                    | Real   | mandatory     | The number density of hydrogen                                                     |
| hydro/iso_sound_speed  | Real   | mandatory     | The isothermal sound speed defined in the `<hydro>` block                          |
| vx_kms                 | Real   | mandatory     | The x velocity in km/s                                                             |
| init_H                 | Real   | 0.0           | The initial abundance of hydrogen gas                                              |
| gaussian_mean          | Real   | 0.5           | The mean of the initial Gaussian profile                                           |
| gaussian_std           | Real   | 0.1           | The standard deviation of the initial Gaussian profile                             |

#### Initial and Final States of the H2 Advection Test Compared to the Analytical Solution with `nx=128`

![H2 Advection H](uploads/chemistry/H2_advection_H.svg)
![H2 Advection H2](uploads/chemistry/H2_advection_H2.svg)

### GOW17 Uniform

This test problem has a uniform hydrodynamic state and evolves the GOW17 network to chemical and thermal equilibrium. It is used in tests of the GOW17 network, and its problem generator also sets up the static PDR test below. Details can be found in Section 4.1 of [Gong et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...42G/abstract).

- Input File: [GOW17_uniform.athinput](../../blob/main-chemistry/inputs/chemistry/GOW17_uniform.athinput)
- `pgen_name`: GOW17_uniform

Problem generator parameters, all are within the `<problem>` block unless otherwise specified:

| Option                 | Type   | Default       | Description                                                                                                  |
| ---------------------- | ------ | ------------- | ------------------------------------------------------------------------------------------------------------ |
| n_H                    | Real   | mandatory     | The number density of hydrogen nuclei in cm$^{-3}$                                                           |
| hydro/iso_sound_speed  | Real   | mandatory     | Sets the initial pressure, $n_\mathrm{H} c_s^2$, in code units. It is defined in the `<hydro>` block          |
| vx_kms                 | Real   | 0.0           | The x velocity in km/s                                                                                       |
| init_default           | Real   | 0.0           | The initial abundance of every species not set with `init_<species>`                                          |
| init_<species>         | Real   | init_default  | The initial abundance of a species, e.g. `init_H2` or `init_CO`, using the species names of the GOW17 network |

### Static PDR

A static photodissociation region (PDR): a uniform 1D slab with $n_\mathrm{H} = 100\ \mathrm{cm^{-3}}$, lit from the inner x1 boundary by a radiation field of $G_0 = 6$ with six-ray radiation. The hydro is frozen with `evolution = kinematic`, the `advect` Riemann solver, and zero velocity, so only the chemistry and radiation evolve. It is a port of the Athena++ static PDR test, see Section 4.2.1 of [Gong et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...42G/abstract), and uses the GOW17 Uniform problem generator.

- Input File: [PDR_static.athinput](../../blob/main-chemistry/inputs/chemistry/PDR_static.athinput)
- `pgen_name`: GOW17_uniform

Notes on the input file:

- Athena++ freezes the hydro with `active = fixed` and takes 100 timesteps of `tlim/nlim`. Frozen hydro places no limit on the AthenaK timestep, so the input sets `dt_max = 100` in `<time>` to take the same 100 timesteps. The radiation is recomputed each timestep, so the result depends on the number of timesteps.
- `kokkos_BDF_atol = 1e-12` instead of Athena++'s 1e-15, see the [Kokkos BDF](ODE-Solvers#kokkos-bdf) notes.
- The mesh is split into 10 MeshBlocks, since MeshBlocks of more than about 6000 cells in x1 do not fit in GPU shared memory for the hydro update.

With these settings, AthenaK agrees with the Athena++ results for this test to a relative error of about $3\times10^{-5}$ in every species and the temperature.

### Moving PDR

A cold slab, $n_\mathrm{H} = 100\ \mathrm{cm^{-3}}$ at 40 K between $x = 5$ and 45 pc, surrounded by warm gas, with everything moving at `vx_kms`. Six-ray radiation of $G_0 = 6$ enters through both x1 boundaries. Both phases start with the same internal energy density, so they start in pressure equilibrium. The test checks that the chemistry is advected correctly: the moving slab should match a static one shifted by the distance it has moved. See Section 4.2.2 of [Gong et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...42G/abstract).

- Input File: [PDR_moving.athinput](../../blob/main-chemistry/inputs/chemistry/PDR_moving.athinput)
- `pgen_name`: PDR_moving

Problem generator parameters, all are within the `<problem>` block unless otherwise specified:

| Option                 | Type   | Default       | Description                                                                                                  |
| ---------------------- | ------ | ------------- | ------------------------------------------------------------------------------------------------------------ |
| vx_kms                 | Real   | mandatory     | The x velocity of all the gas in km/s                                                                        |
| n_H_warm               | Real   | 0.1           | The number density of hydrogen nuclei of the warm gas in cm$^{-3}$                                           |
| init_default           | Real   | 0.0           | The initial abundance of every species not set with `init_<species>`, in both phases                          |
| init_<species>         | Real   | init_default  | The initial abundance of a species, e.g. `init_H2`, using the species names of the GOW17 network              |

The cold slab's density, temperature, and position are fixed, as in Athena++.

The provided input file is a reduced resolution version with heating and cooling on, which runs quickly. To reproduce the results in Gong et al. (2023), use 13750 cells (with MeshBlocks of at most about 6000 cells in x1 on GPUs) and `cfl_number = 0.3`. Also add the following, which match the paper's constant $\mu = 2.33$ with zero heating and cooling rates:

```
<hydro>
iso_sound_speed  = 0.376068874   # with GOW17_mu_iso sets the chemistry temperature, 40 K

<chemistry>
GOW17_isothermal = true
GOW17_mu_iso     = 2.33333333333
```

With heating and cooling on, the warm gas cools quickly and the slab expands into it, so the results differ from the paper.

### NCR Equilibrium

The thermal and chemical equilibrium test of the NCR network, from Section 7.1 of [Kim et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..264...10K/abstract) and a port of the TIGRESS-NCR `photchem_equil` problem. It is a 1D row of 161 independent one-zone problems with constant radiation, $G_0 = 1$ and $\xi_\mathrm{CR} = 2\times10^{-16}\ \mathrm{s^{-1}}$. Every cell starts at rest with $x_\mathrm{H^+} = 0.01$ and $x_\mathrm{H_2} = 0$. The hydro is frozen with `evolution = kinematic`, the `advect` Riemann solver, and zero velocity. Unlike TIGRESS there is no equilibrium pre-solve: the network is integrated until it reaches equilibrium, with `tlim` and `dt_max` in `<time>` setting how long and in how many steps.

- `iprob = 0`: thermal and chemical equilibrium as a function of density. $n_\mathrm{H}$ is log-spaced from $10^{-2}$ to $10^6\ \mathrm{cm^{-3}}$ and the gas starts at $P = 1.4\, n_\mathrm{H} k_B \times 10^3$ K, TIGRESS's $T/\mu = 10^3$ K.
- `iprob = 1`: chemical equilibrium as a function of temperature. $T$ is log-spaced from 10 to $10^8$ K at $n_\mathrm{H} = 1\ \mathrm{cm^{-3}}$ and held fixed with `NCR_chem_only = true`. The initial internal energy is set from the target $T$ with $x_e$ in equilibrium at that $T$.

The position along each grid comes from the global cell index, so any MeshBlock or MPI decomposition gives the same problem.

Near $n_\mathrm{H} = 10^{5.65}\ \mathrm{cm^{-3}}$ the `iprob = 0` problem has no stable equilibrium. For $G_0 = 1$, $Z = 1$, and $\xi_\mathrm{CR} = 2\times10^{-16}\ \mathrm{s^{-1}}$ the equilibrium is Hopf-unstable for $\log_{10} n_\mathrm{H} \approx 5.647$–$5.663$: at $\log_{10} n_\mathrm{H} = 5.65$, with TIGRESS's default `dvdr`, the Jacobian of the NCR equations at the equilibrium has the eigenvalues $+155 \pm 3405i$ per code time unit, so the gas oscillates about it with a growing amplitude instead of settling. The cause is H2 UV-pumping heating, which through its temperature-dependent critical density grows as $T^{0.93}$, faster than the OI cooling that balances it, $\propto T^{0.62\text{–}0.73}$. The cell at $\log_{10} n_\mathrm{H} = 5.65$ of the default 161-cell grid lies in this window, and BDF takes a very large number of steps there. TIGRESS hides this by capping its equilibrium iterations, so its own reference at $\log_{10} n_\mathrm{H} = 5.65$ is an unconverged snapshot.

- Input File: [NCR_equilibrium.athinput](../../blob/main-chemistry/inputs/chemistry/NCR_equilibrium.athinput)
- `pgen_name`: NCR_equilibrium

Problem generator parameters, all are within the `<problem>` block unless otherwise specified:

| Option                      | Type | Default   | Description                                                                                    |
| --------------------------- | ---- | --------- | ---------------------------------------------------------------------------------------------- |
| iprob                       | int  | mandatory | 0 for equilibrium as a function of $n_\mathrm{H}$, 1 for chemical equilibrium as a function of $T$ |
| log_nH_min                  | Real | -2.0      | `iprob = 0`: $\log_{10} n_\mathrm{H}$ of the first cell, in cm$^{-3}$                          |
| log_nH_max                  | Real | 6.0       | `iprob = 0`: $\log_{10} n_\mathrm{H}$ of the last cell, in cm$^{-3}$                           |
| nH                          | Real | 1.0       | `iprob = 1`: $n_\mathrm{H}$ in cm$^{-3}$                                                       |
| log_tgas_min                | Real | 1.0       | `iprob = 1`: $\log_{10} T$ of the first cell, in K                                             |
| log_tgas_max                | Real | 8.0       | `iprob = 1`: $\log_{10} T$ of the last cell, in K                                              |
| chemistry/network           | string | mandatory | Must be `NCR`                                                                                |
| chemistry/NCR_chem_only     | bool | false     | Must be true for `iprob = 1`                                                                   |
| chemistry/radiation_integrator | string | constant | Must be `constant` for `iprob = 1`. `iprob = 0` also runs with `six_ray`, e.g. lit from one face with `radiation_G0_inner_x1`, though there is no reference solution for it |

The tab output uses `data_format = %24.16e`, which is needed to recover $x_\mathrm{HI} \approx 10^{-10}$ in hot gas and $x_e$ from the output.

The automated test, `tst/test_suite/chemistry/test_NCR_gpu.py`, is a port of TIGRESS-NCR's `equil.py`. It runs both problems for every `NCR_interp_mode` and for $Z_g = Z_d$ = 0.01, 0.1, 1, and $10^{0.5}$, and compares them with the TIGRESS-NCR reference solutions in `tst/test_suite/chemistry/data/ncr_equilibrium.npz`:

- `iprob = 0`: $P/k_B$, $x_\mathrm{H^+}$, and $x_\mathrm{H_2}$ against the unshielded reference solutions, to a relative error of $10^{-2}$, for $n_\mathrm{H} < 10^3\ \mathrm{cm^{-3}}$, the same rows that TIGRESS's `equil.py` compares. The test runs only these 100 cells, $n_\mathrm{H} = 10^{-2}$ to $10^{2.95}\ \mathrm{cm^{-3}}$ in steps of 0.05 dex, which also keeps it clear of the unstable window above.
- `iprob = 1`: $x_\mathrm{HI}$, $x_\mathrm{H_2}$, and $x_e$ against TIGRESS's exact-mode solution from a converged rerun of TIGRESS-NCR (commit `118ffe56acd7712ace476e1e687a234a96fd5577`). TIGRESS's stored references stop the equilibrium iteration once the elapsed time reaches the problem generator's $dt = 10^5$, while the lagged-$x_e$ iteration is still moving, which leaves $x_e$ off by up to $5$–$8\times10^{-4}$ at $Z \ge 1$, $4\times10^{-3}$ at $Z = 0.1$, and $10^{-2}$ at $Z = 0.01$. The rerun drops that stop and iterates until the per-iteration relative changes of $x_\mathrm{H^+}$, $x_e$, and $x_\mathrm{HI}$ are below $10^{-13}$, with the physics unchanged; the test's docstring has the details. The tolerances are TIGRESS's per-mode relative tolerances: $2$–$3\times10^{-4}$ for exact and log-log, $5\times10^{-3}$ for NQT2, and $2\times10^{-2}$ for NQT1. $x_e$ is not an output, so it is recovered from the pressure, $P = n_\mathrm{H} k_B T (1.1 + x_e - x_\mathrm{H_2})$. Above `NCR_temp_hot1`, where TIGRESS sets $x_\mathrm{H_2}$ to zero, $x_\mathrm{H_2}$ is only checked to be below $5\times10^{-14}$.

The CPU and MPI tests run the log-log mode at $Z = 1$ only.
