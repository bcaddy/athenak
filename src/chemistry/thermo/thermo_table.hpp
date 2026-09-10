#ifndef CHEMISTRY_THERMO_THERMO_TABLE_HPP_
#define CHEMISTRY_THERMO_THERMO_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file thermo_table.hpp
//  \brief Tabulated temperature-only coefficients for the fine-structure line
//  cooling, on a shared log-T grid.

#include "athena.hpp"
#include "chemistry/thermo/thermo.hpp"

namespace chemistry {

/*!
 * \brief Collisional rate coefficients and Boltzmann factors for the CII, CI
 * and OI fine-structure lines, tabulated against temperature.
 *
 * \details The fine-structure coolers split cleanly. Their collisional rate
 * coefficients are pure functions of T -- mostly of the form
 * `k = A * (T/100)^(a + b ln(T/100))`, a Gaussian in log T -- and the
 * abundances enter only afterwards, linearly, as `q10 = k10HI*nHI + k10H2*nH2 +
 * k10e*ne`. The level-population solvers those feed (`Cooling2Level_`,
 * `Cooling3Level_`) contain no transcendentals at all.
 *
 * So the entire transcendental cost of these three coolers is a function of one
 * variable. Measured on the GOW17 network, `CoolingCI`, `CoolingOI` and
 * `CoolingCII` together account for 35 of the 80 transcendental calls in one
 * heating/cooling evaluation, and the thermodynamics as a whole is 86% of the
 * chemistry solver's cost.
 *
 * Tabulating on a *shared* grid is what makes it cheap: `log10(T)`, the cell
 * index and the interpolation weight are computed once, and each of the 28
 * coefficients is then two loads and a lerp.
 *
 * The ortho/para H2 mixing collapses into the table as well: `fo_` and `fp_` are
 * constants, so `k10H2 = k10H2o*fo_ + k10H2p*fp_` is itself a pure function of
 * T and six `pow` calls become three table entries.
 */
struct ThermoTable {
  // Column layout. Keep in step with Fill() below.
  enum : int {
    // CII, from q10CII_
    ICII_k10e = 0, ICII_k10HI, ICII_k10H2, ICII_boltz,
    // CI
    ICI_k10e, ICI_k20e, ICI_k21e,
    ICI_k10HI, ICI_k20HI, ICI_k21HI,
    ICI_k10H2, ICI_k20H2, ICI_k21H2,
    ICI_boltz10, ICI_boltz20, ICI_boltz21,
    // OI
    IOI_k10e, IOI_k20e, IOI_k21e,
    IOI_k10HI, IOI_k20HI, IOI_k21HI,
    IOI_k10H2, IOI_k20H2, IOI_k21H2,
    IOI_boltz10, IOI_boltz20, IOI_boltz21,
    // Lyman alpha: both the prefactor and its Boltzmann factor are pure T
    ILYA_fac, ILYA_k01,
    // H2 formation and UV pumping heating share these two, and the two
    // functions are otherwise identical bar their return line -- see the NOTE
    // above HeatingH2gr. Tabulating them removes the duplicate evaluation.
    IH2_geffH, IH2_geffH2,
    NCOEF
  };

  /// Grid in log10(T). The lower edge is below any temperature the cooling is
  /// evaluated at (CoolingTerm returns zero under temperature_min_cooling) and
  /// the upper edge is above temperature_max_cooling_nm, where the
  /// fine-structure coolers are capped anyway.
  static constexpr int n_T = 512;
  static constexpr Real logT_min = 0.0;   // 1 K
  static constexpr Real logT_max = 5.0;   // 1e5 K
  static constexpr Real dlogT = (logT_max - logT_min) / (n_T - 1);
  static constexpr Real inv_dlogT = (n_T - 1) / (logT_max - logT_min);

  /// (n_T, NCOEF), device resident.
  DvceArray2D<Real> data;

  /// Interpolation weights for one temperature: the two rows and the linear
  /// weight between them. Computed once per evaluation and reused for every
  /// coefficient, which is the whole point of the shared grid.
  struct Slot {
    int i0;
    int i1;
    Real w;
  };

  static KOKKOS_INLINE_FUNCTION Slot Locate(const Real T) {
    const Real u = (Kokkos::log10(Kokkos::fmax(T, 1.0)) - logT_min) * inv_dlogT;
    const Real uc = Kokkos::fmin(Kokkos::fmax(u, 0.0),
                                 static_cast<Real>(n_T - 1) - 1.0e-9);
    const int i0 = static_cast<int>(uc);
    return Slot{i0, i0 + 1, uc - static_cast<Real>(i0)};
  }

  KOKKOS_INLINE_FUNCTION Real At(const Slot& s, const int c) const {
    return (1.0 - s.w) * data(s.i0, c) + s.w * data(s.i1, c);
  }
};

}  // namespace chemistry

#endif  // CHEMISTRY_THERMO_THERMO_TABLE_HPP_
