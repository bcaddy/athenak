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

#include <cstdint>
#include <cstring>

#include "athena.hpp"
#include "chemistry/thermo/thermo.hpp"

namespace chemistry {

/*!
 * \brief Collisional rate coefficients and Boltzmann factors for the CII, CI
 * and OI fine-structure lines, tabulated against temperature.
 *
 * \details The collisional rate coefficients of the fine-structure coolers are
 * functions of T alone -- mostly `k = A * (T/100)^(a + b ln(T/100))` -- and the
 * abundances enter only afterwards, linearly, as `q10 = k10HI*nHI + k10H2*nH2 +
 * k10e*ne`. The level-population solvers they feed (`Cooling2Level_`,
 * `Cooling3Level_`) contain no transcendentals, so tabulating against T removes
 * the whole transcendental cost of those coolers.
 *
 * The grid is shared across every coefficient: `log10(T)`, the cell index and
 * the interpolation weight are computed once per evaluation and each coefficient
 * is then two loads and a lerp. Layout is T-major, so all coefficients at one
 * temperature are contiguous.
 */
struct ThermoTable {
  // Column layout. Keep in step with BuildThermoTable.
  enum : int {
    // CII
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
    // Lyman alpha: prefactor and Boltzmann factor
    ILYA_fac, ILYA_k01,
    // H2 formation and UV pumping heating, which share both
    IH2_geffH, IH2_geffH2,
    // H2 rovibrational line cooling: per-partner low-density coefficients and
    // the LTE rate
    IH2C_LHI, IH2C_LH2, IH2C_LHe, IH2C_LHplus, IH2C_Le, IH2C_LTE,
    /// The grid temperature itself. Reading it from each bracketing row gives
    /// the two temperatures a row pair straddles, which is the denominator of
    /// a temperature derivative taken by differencing the rows.
    ITEMP,
    NCOEF
  };

  /// Grid in log10(T). The lower edge is below any temperature the cooling is
  /// evaluated at (CoolingTerm returns zero under temperature_min_cooling) and
  /// the upper edge is above temperature_max_cooling_nm, where the
  /// fine-structure coolers are capped anyway.
  /// Resolution follows the cold-range grid of TIGRESS-NCR (ncr_rates.hpp,
  /// kDLogTCold), 0.004 dex. The lower edge is GOW17_temperature_min_cooling's
  /// default rather than NCR's 10 K, since CoolingTerm is evaluated down to 1 K.
  static constexpr int n_T = 1251;
  static constexpr Real logT_min = 0.0;   // 1 K
  static constexpr Real logT_max = 5.0;   // 1e5 K
  static constexpr Real dlogT = (logT_max - logT_min) / (n_T - 1);
  static constexpr Real inv_dlogT = (n_T - 1) / (logT_max - logT_min);
  /// Top of the grid in K. Coefficients that are still varying above it must
  /// fall back to their analytic form rather than clamp.
  static constexpr Real T_max = 1.0e5;

  /// Coefficients that vanish (H2 line cooling below its cutoff) cannot be
  /// stored as a logarithm, so they are floored here instead. Small enough that
  /// a floored coefficient contributes nothing to any sum it appears in.
  static constexpr Real value_floor = 1.0e-100;

  /// (n_T, NCOEF), device resident.
  DvceArray2D<Real> data;

  /// Grid origin and inverse spacing in nqt1_log space. Not constexpr because
  /// nqt1_log is not; filled by BuildThermoTable.
  Real nqt_t_min;
  Real nqt_idt;

  // NQT (Not Quite Transcendental) log and exp, Hammond et al. 2025, ApJS
  // 277:65. nqt1 is a piecewise-linear approximant of log2 built from the
  // IEEE-754 exponent and mantissa; nqt2 adds a quadratic correction, is
  // C1-continuous, and holds ~0.1% maximum error. The grid is indexed with
  // nqt1 and the values are stored under nqt2, matching NCR's kNqt2 mode.
  static KOKKOS_INLINE_FUNCTION double nqt1_log(const double x) {
    std::int64_t i;
    std::memcpy(&i, &x, sizeof(double));
    // as_int(1.0) = 4607182418800017408, as_int(2.0) - as_int(1.0) = 2^52
    return static_cast<double>(i - 4607182418800017408LL) / 4503599627370496.0;
  }

  /// Inverse of nqt1_log. Used only to place the grid points.
  static KOKKOS_INLINE_FUNCTION double nqt1_exp(const double y) {
    std::int64_t i = static_cast<std::int64_t>(y * 4503599627370496.0) +
                     4607182418800017408LL;
    double x;
    std::memcpy(&x, &i, sizeof(double));
    return x;
  }

  static KOKKOS_INLINE_FUNCTION double nqt2_log(const double x) {
    std::int64_t i;
    std::memcpy(&i, &x, sizeof(double));
    // fractional mantissa in [0,1)
    const double b =
        static_cast<double>(i & 0x000FFFFFFFFFFFFFLL) / 4503599627370496.0;
    return static_cast<double>(i - 4607182418800017408LL) / 4503599627370496.0 +
           b * (1.0 - b) / 3.0;
  }

  static KOKKOS_INLINE_FUNCTION double nqt2_exp(const double y) {
    // A true floor, not truncation: floored coefficients sit near y = -332,
    // where truncating toward zero gives a negative fractional part and the
    // square root below returns NaN.
    const std::int64_t e = static_cast<std::int64_t>(Kokkos::floor(y));
    const double f = y - static_cast<double>(e);
    // Invert b + b*(1-b)/3 = f, i.e. b^2 - 4b + 3f = 0.
    const double b = 2.0 - Kokkos::sqrt(4.0 - 3.0 * f);
    const std::int64_t mantissa =
        static_cast<std::int64_t>(b * 4503599627370496.0);
    std::int64_t i = ((e + 1023LL) << 52) | mantissa;
    double x;
    std::memcpy(&x, &i, sizeof(double));
    return x;
  }

  /// Interpolation weights for one temperature: the two rows and the linear
  /// weight between them. Computed once per evaluation and reused for every
  /// coefficient, which is the whole point of the shared grid.
  struct Slot {
    int i0;
    int i1;
    Real w;
  };

  KOKKOS_INLINE_FUNCTION Slot Locate(const Real T) const {
    const Real u = (nqt1_log(Kokkos::fmax(T, 1.0)) - nqt_t_min) * nqt_idt;
    const Real uc = Kokkos::fmin(Kokkos::fmax(u, 0.0),
                                 static_cast<Real>(n_T - 1) - 1.0e-9);
    const int i0 = static_cast<int>(uc);
    return Slot{i0, i0 + 1, uc - static_cast<Real>(i0)};
  }

  /// Coefficients are stored and interpolated linearly, not as logarithms.
  /// A log-log table would be more accurate for these power-law rates, but it
  /// costs a decode per coefficient, and Edot reads about thirty of them per
  /// call against the single index computed by Locate. At 0.004 dex the linear
  /// interpolation error is ~1e-4 for a rate of index 2, well inside the
  /// accuracy this module targets.
  KOKKOS_INLINE_FUNCTION Real At(const Slot& s, const int c) const {
    const Real v0 = data(s.i0, c);
    return v0 + s.w * (data(s.i1, c) - v0);
  }
};

/*!
 * \brief Fill the table on the host and copy it to the device.
 *
 * \details Evaluates Thermo::CIIRates, CIRates and OIRates at each grid point,
 * the same helpers the analytic path calls. Call once per process.
 */
void BuildThermoTable(ThermoTable& tab);

}  // namespace chemistry

#endif  // CHEMISTRY_THERMO_THERMO_TABLE_HPP_
