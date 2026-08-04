# Equations of State for Dynamical GRMHD
AthenaK uses a custom version of the [PrimitiveSolver](https://github.com/jfields7/primitive-solver) framework for equations of state for GRMHD in dynamical spacetimes. There is currently support for the following EOS models:
* Ideal gas
* Piecewise-polytropic EOS with gamma-law thermal extension
* 1D tabulated nuclear EOS in the [PyCompOSE](https://github.com/computationalrelativity/PyCompOSE) format with gamma-law thermal extension
* 3D tabulated nuclear EOS in the [PyCompOSE](https://github.com/computationalrelativity/PyCompOSE) format

All equations of state are written in terms of the baryon number density $n$, temperature $T$, and particle species fractions $\mathbf{Y}$ (if supported by the model) and assume units where the Boltzmann constant $k_B = 1$.

## Ideal gases
An EOS of the form
```math
\begin{equation}
\begin{aligned}
  P &= n T \\
    &= \left(\Gamma - 1\right)\rho \epsilon.
\end{aligned}
\end{equation}
```

It can be enabled by setting `mhd/eos_policy=ideal` in the parameter file. The adiabatic index, $\Gamma$, is set via `mhd/gamma`. To ensure that the sound speed is physical, AthenaK enforces $1 < \Gamma \leq 2$. This is the simplest and fastest EOS supported by `DynGRMHD`, but it is not usually very physically accurate for most problems of interest.

## Piecewise polytropes
An EOS of the form
```math
\begin{align}
  P &= P_i \left(\frac{n}{n_i}\right)^{\Gamma_i} + n T, \\
  e &= m_b n \left(1 + \epsilon_i\right) + \frac{P_i}{\Gamma_i - 1}\left(\frac{n}{n_i}\right)^{\Gamma_i} + \frac{n T}{\Gamma_\mathrm{th} - 1},
\end{align}
```
where $P_i$ and $\epsilon_i$ are the cold (zero-temperature) pressure and specific internal energy at $n_i$, $\Gamma_i$ is the polytropic index between $n_{i-1}$ and $n_i$, and $\Gamma_\mathrm{th}$ is the adiabatic index for the thermal extension.

Piecewise polytropes can be enabled by setting `mhd/eos_policy=piecewise_poly` and use the conventions in [RePrimAnd](https://wokast.github.io/RePrimAnd/eos_barotr_available.html#piecewise-polytropic-eos). The following parameters must be defined:
* `mhd/pwp_poly_rmd`: a reference mass density in kg/m^3 used to determine P_0.
* `mhd/pwp_density_pieces_<#>`: a zero-indexed dividing mass density $\rho_i$ in kg/m^3. Each piece must be strictly increasing, i.e., `pwp_density_pieces_0 < pwp_density_pieces_1`, etc.
* `mhd/pwp_gamma_pieces_<#>`: a zero-indexed polytropic index $\Gamma_i$. It is incumbent on the user to ensure that $\Gamma_i$ cannot produce superluminal sound speeds over its density range.
* `mhd/gamma_thermal`: the thermal adiabatic index $\Gamma_\mathrm{th}$

$P_i$, $n_i$, and $\epsilon_i$ can be determined uniquely from these parameters. Note that there is a hard limit of seven total pieces fixed by a compile-time constant. This could be altered if necessary, but a hybrid tabulated EOS will likely be more efficient for a large number of pieces.

## Tabulated EOS
An EOS determined by a lookup table. Any 1D or 3D EOS in the [CompOSE](https://compose.obspm.fr/) database can be used. These can be converted into the `.athtab` files readable by AthenaK using the [PyCompOSE](https://github.com/https://github.com/computationalrelativity/PyCompOSE) library.

### Hybrid EOS
A 1D hybrid EOS is enabled using `mhd/eos_policy=hybrid`. It expects a table which is evenly spaced in $\log n$ and extends it to finite temperatures with a thermal gamma law, as in the piecewise polytrope case. Thus it may be written as
```math
\begin{align}
P(n, T) &= P_c(n) + n T, \\
e(n, T) &= e_c(n) + \frac{n T}{\Gamma_\mathrm{th} - 1}
\end{align}
```
for a zero-temperature pressure, $P_c(n)$, and zero-temperature energy, $e_c(n)$, which are interpolated from the table. The table file is supplied via the parameter `mhd/table=<filename>`, and $\Gamma_\mathrm{th}$ is set using `mhd/gamma_thermal`.

### Full microphysical table
The 3D EOS is enabled with `eos_policy=compose` and takes a table which is evenly spaced in $\log n$, charge fraction $Y_q$, and $\log T$. Thermodynamic quantities are computed by performing lookup operations and linearly interpolating between the nearest points. The table file is supplied via the parameter `mhd/table=<filename>`. The EOS will be as realistic as the input physics used to design the table, though it is usually also the most computationally expensive.

### Tables with not-quite-transcendental functions
Both tabulated EOS policies also support tables scaled using "not-quite-transcendental" (NQT) functions (see *ApJS* 277 (2025) 2, 65, [2501.05410 [physics.comp-ph]](https://arxiv.org/abs/2501.05410)) in place of logarithms and exponents. NQT functions are second-order approximations to $\log_2 x$ and $2^x$ which can be written using floating-point bit hacks to provide a modest performance improvement (usually around 20% or so) on most architectures. Note that though these functions are "approximations", they are exactly invertible, so an EOS retabulated such that $n$ and $T$ are linear in NQT-space is still exact in that space. The approximation is that the scaling is now only approximately logarithmic rather than truly logarithmic. This can be tricky if one needs to perform finite-difference operations on the table, since $d\log_{NQT} A/d\log_{NQT} B \neq (B/A) dA/dB$, but in practice this is not an issue in AthenaK.

PyCompOSE has support for generating NQT-scaled tables (see [this example](https://github.com/computationalrelativity/PyCompOSE/blob/master/examples/SFHo_NQT.py)), which AthenaK can load as it would any other table by setting `mhd/use_NQT=true`.

__Finally, a warning__: because NQT functions are implemented using bit hacks for maximum performance, they make assumptions about floating-point layouts and are not truly portable. Extremely aggressive compiler optimizations (e.g., GCC's `-ffast-math` and similar flags), big-endian layouts, and architectures which are not compliant with the IEE 754 floating-point standard can all break NQT functions. To test if your machine configuration is compatible, try running the `unit_tests/eos_compose_test` pgen with an NQT-scaled 3D table.

## A note on units
`PrimitiveSolver` automatically handles unit conversions between code units and EOS units. For an ideal gas, the baryon mass $m_b = 1$ by default, so $\rho = n$ and $T$ is dimensionless. Physical temperatures can theoretically be recovered by assuming a value for the baryon mass, but these are often not meaningful for neutron star simulations. Input parameters for piecewise polytropes are in SI units for consistency with other codes, but these are converted into a nearly dimensionless form similar to the ideal gas case during initialization.

All tabulated equations of state assume standard nuclear units with $c=k_B=1$. The length unit is in $\mathrm{fm}$, and energies and temperatures are measured in $\mathrm{MeV}$. See the [CompOSE manual](https://compose.obspm.fr/manual/) for more information.

For `DynGRMHD`, the code units are usually chosen as geometric solar units with $G=c=\mathrm{M}_\odot=1$. Temperatures are usually left unconverted, since they do not directly enter any calculation except through the EOS. Therefore, the code temperature when using an ideal gas or piecewise polytrope will be dimensionless, and the temperature when using a tabulated EOS will be in $\mathrm{MeV}$.