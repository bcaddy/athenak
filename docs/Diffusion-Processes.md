Diffusion processes implemented in `AthenaK` include:
- Hydrodynamic diffusion: fluid viscosity and thermal conduction
- Field diffusion: Ohmic diffusion

# Input File

Diffusive physics such as viscosity and/or thermal conduction can be constructed in `<hydro>` or `<mhd>`. Ohmic diffusion is constructed in `<mhd>` block. They are enrolled in the corresponding `<block>` in the input file:
```
<hydro>
viscosity   = 1.0e-3   # coefficient of isotropic shear viscosity
conductivity = 1.0e-3  # coefficient of isotropic thermal conduction
```
or
```
<mhd>
viscosity   = 1.0e-3     # coefficient of isotropic shear viscosity
conductivity = 1.0e-3    # coefficient of isotropic thermal conduction
ohmic_resistivity = 0.01 # coefficient of Ohmic resistivity
```

# Thermal conduction 

The heat flux $`\boldsymbol{Q}`$ is defined by
```math
    \boldsymbol{Q}=-\mathcal{K}\nabla T,
```
where the coefficient $`\mathcal{K}`$ is the conductivity. The current implementation uses a dimensionless system of units (see [Units](https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenak/-/wikis/Units)) in that the factor $`(\bar{m}/k_{\rm B})`$ is not included in calculating the temperature. Instead, $`T=P/\rho`$ is adopted. Users must use dimensionless conductivity as the input parameter.

Note that the coefficient $`\mathcal{K}`$ corresponds to conductivity, **not** a diffusivity. This is different from the coefficient used in `Athena++`.
