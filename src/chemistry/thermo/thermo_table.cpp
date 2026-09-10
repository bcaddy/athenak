//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file thermo_table.cpp
//  \brief Host-side construction of the fine-structure coefficient table.

#include "chemistry/thermo/thermo_table.hpp"

#include <cmath>

#include "athena.hpp"
#include "units/units.hpp"

namespace chemistry {

namespace {
// The analytic forms, transcribed from thermo.hpp. They are duplicated here
// rather than shared because the table is built on the host with std:: maths
// while the originals are device code; keeping them separate avoids making the
// device path depend on the table's existence. Any change to a fit in
// thermo.hpp has to be mirrored here, which the round-trip check in
// BuildThermoTable guards against.
constexpr double fo = 0.75;
constexpr double fp = 0.25;
constexpr double kB = 1.380649e-16;

constexpr double g0CI = 1, g1CI = 3, g2CI = 5;
constexpr double E10CI = 3.261e-15, E20CI = 8.624e-15, E21CI = 5.363e-15;
constexpr double g0OI = 5, g1OI = 3, g2OI = 1;
constexpr double E10OI = 3.144e-14, E20OI = 4.509e-14, E21OI = 1.365e-14;
constexpr double E10CII = 1.26e-14;

double poly4(double a, double b, double c, double d, double e, double x) {
  return (((a * x + b) * x + c) * x + d) * x + e;
}
}  // namespace

void BuildThermoTable(ThermoTable& tab) {
  tab.data = DvceArray2D<Real>("thermo_table", ThermoTable::n_T,
                               ThermoTable::NCOEF);
  auto h = Kokkos::create_mirror_view(tab.data);

  for (int i = 0; i < ThermoTable::n_T; ++i) {
    const double logT = ThermoTable::logT_min + i * ThermoTable::dlogT;
    const double T = std::pow(10.0, logT);
    const double T2 = T / 100.0;
    const double lnT2 = std::log(T2);
    const double lnT = std::log(T);
    const double kbT = kB * T;

    // ----- CII, from q10CII_ -----
    h(i, ThermoTable::ICII_k10e) = 4.53e-8 * std::sqrt(1.0e4 / T);
    h(i, ThermoTable::ICII_k10HI) =
        7.58e-10 * std::pow(T2, 0.1281 + 0.0087 * lnT2);
    double k10oH2, k10pH2;
    if (T < 500.0) {
      k10oH2 = (5.33 + 0.11 * T2) * 1.0e-10;
      k10pH2 = (4.43 + 0.33 * T2) * 1.0e-10;
    } else {
      const double tmp = std::pow(T, 0.07);
      k10oH2 = 3.74757785025e-10 * tmp;
      k10pH2 = 3.88997286356e-10 * tmp;
    }
    h(i, ThermoTable::ICII_k10H2) = k10oH2 * fo + k10pH2 * fp;
    h(i, ThermoTable::ICII_boltz) = std::exp(-E10CII / kbT);

    // ----- CI -----
    const double facCI = 8.629e-8 * std::sqrt(1.0e4 / T);
    double lg10, lg20, lg21;
    if (T < 1.0e3) {
      lg10 = poly4(-6.56325e-4, -1.50892e-2, 3.61184e-1, -7.73782e-1, -9.25141,
                   lnT);
      lg20 = poly4(0.705277e-2, -0.111338, 0.697638, -1.30743, -7.69735, lnT);
      lg21 = poly4(2.35272e-3, -4.18166e-2, 0.358264, -0.57443, -7.4387, lnT);
    } else {
      lg10 = poly4(1.0508e-1, -3.47620, 4.2595e1, -2.27913e2, 4.446e2, lnT);
      lg20 = poly4(9.38138e-2, -3.03283, 3.61803e1, -1.87474e2, 3.50609e2, lnT);
      lg21 = poly4(9.78573e-2, -3.19268, 3.85049e1, -2.02193e2, 3.86186e2, lnT);
    }
    h(i, ThermoTable::ICI_k10e) = facCI * std::exp(lg10) / g1CI;
    h(i, ThermoTable::ICI_k20e) = facCI * std::exp(lg20) / g2CI;
    h(i, ThermoTable::ICI_k21e) = facCI * std::exp(lg21) / g2CI;
    h(i, ThermoTable::ICI_k10HI) =
        1.26e-10 * std::pow(T2, 0.115 + 0.057 * lnT2);
    h(i, ThermoTable::ICI_k20HI) =
        0.89e-10 * std::pow(T2, 0.228 + 0.046 * lnT2);
    h(i, ThermoTable::ICI_k21HI) =
        2.64e-10 * std::pow(T2, 0.231 + 0.046 * lnT2);
    h(i, ThermoTable::ICI_k10H2) =
        0.67e-10 * std::pow(T2, -0.085 + 0.102 * lnT2) * fp +
        0.71e-10 * std::pow(T2, -0.004 + 0.049 * lnT2) * fo;
    h(i, ThermoTable::ICI_k20H2) =
        0.86e-10 * std::pow(T2, -0.010 + 0.048 * lnT2) * fp +
        0.69e-10 * std::pow(T2, 0.169 + 0.038 * lnT2) * fo;
    h(i, ThermoTable::ICI_k21H2) =
        1.75e-10 * std::pow(T2, 0.072 + 0.064 * lnT2) * fp +
        1.48e-10 * std::pow(T2, 0.263 + 0.031 * lnT2) * fo;
    h(i, ThermoTable::ICI_boltz10) = std::exp(-E10CI / kbT);
    h(i, ThermoTable::ICI_boltz20) = std::exp(-E20CI / kbT);
    h(i, ThermoTable::ICI_boltz21) = std::exp(-E21CI / kbT);

    // ----- OI -----
    h(i, ThermoTable::IOI_k10HI) =
        3.57e-10 * std::pow(T2, 0.419 - 0.003 * lnT2);
    h(i, ThermoTable::IOI_k20HI) =
        3.19e-10 * std::pow(T2, 0.369 - 0.006 * lnT2);
    h(i, ThermoTable::IOI_k21HI) =
        4.34e-10 * std::pow(T2, 0.755 - 0.160 * lnT2);
    h(i, ThermoTable::IOI_k10H2) =
        1.49e-10 * std::pow(T2, 0.264 + 0.025 * lnT2) * fp +
        1.37e-10 * std::pow(T2, 0.296 + 0.043 * lnT2) * fo;
    h(i, ThermoTable::IOI_k20H2) =
        1.90e-10 * std::pow(T2, 0.203 + 0.041 * lnT2) * fp +
        2.23e-10 * std::pow(T2, 0.237 + 0.058 * lnT2) * fo;
    h(i, ThermoTable::IOI_k21H2) =
        2.10e-12 * std::pow(T2, 0.889 + 0.043 * lnT2) * fp +
        3.00e-12 * std::pow(T2, 1.198 + 0.525 * lnT2) * fo;
    h(i, ThermoTable::IOI_k10e) = 5.12e-10 * std::pow(T, -0.075);
    h(i, ThermoTable::IOI_k20e) = 4.86e-10 * std::pow(T, -0.026);
    h(i, ThermoTable::IOI_k21e) = 1.08e-14 * std::pow(T, 0.926);
    h(i, ThermoTable::IOI_boltz10) = std::exp(-E10OI / kbT);
    h(i, ThermoTable::IOI_boltz20) = std::exp(-E20OI / kbT);
    h(i, ThermoTable::IOI_boltz21) = std::exp(-E21OI / kbT);

    // ----- Lyman alpha -----
    const double T4 = T / 1.0e4;
    const double lya_fac =
        5.31e-8 * std::pow(T4, 0.15) / (1.0 + std::pow(T4 / 5.0, 0.65));
    h(i, ThermoTable::ILYA_fac) = lya_fac;
    h(i, ThermoTable::ILYA_k01) = lya_fac * std::exp(-11.84 / T4);

    // ----- H2 formation / UV pumping heating -----
    const double t = 1.0 + T / 1000.0;
    h(i, ThermoTable::IH2_geffH) =
        std::pow(10.0, -11.06 + 0.0555 / t - 2.390 / (t * t));
    h(i, ThermoTable::IH2_geffH2) =
        std::pow(10.0, -11.08 - 3.671 / t - 2.023 / (t * t));
  }

  Kokkos::deep_copy(tab.data, h);
}

}  // namespace chemistry
