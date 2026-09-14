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
// Level energies, for the Boltzmann factors.
constexpr double kB = 1.380649e-16;
constexpr double E10CI = 3.261e-15, E20CI = 8.624e-15, E21CI = 5.363e-15;
constexpr double E10OI = 3.144e-14, E20OI = 4.509e-14, E21OI = 1.365e-14;
constexpr double E10CII = 1.26e-14;
}  // namespace

void BuildThermoTable(ThermoTable& tab) {
  tab.data = DvceArray2D<Real>("thermo_table", ThermoTable::n_T,
                               ThermoTable::NCOEF);
  auto h = Kokkos::create_mirror_view(tab.data);

  // The grid is uniform in nqt1_log, not in log10, so that Locate needs only
  // bit operations to find a cell.
  tab.nqt_t_min = ThermoTable::nqt1_log(std::pow(10.0, ThermoTable::logT_min));
  const double nqt_t_max =
      ThermoTable::nqt1_log(std::pow(10.0, ThermoTable::logT_max));
  tab.nqt_idt = (ThermoTable::n_T - 1) / (nqt_t_max - tab.nqt_t_min);

  for (int i = 0; i < ThermoTable::n_T; ++i) {
    const double T =
        ThermoTable::nqt1_exp(tab.nqt_t_min + i / tab.nqt_idt);
    const double kbT = kB * T;

    const Thermo::CIICoefs kcii = Thermo::CIIRates(T);
    const Thermo::CICoefs kci = Thermo::CIRates(T);
    const Thermo::OICoefs koi = Thermo::OIRates(T);

    // ----- CII -----
    h(i, ThermoTable::ICII_k10e) = kcii.k10e;
    h(i, ThermoTable::ICII_k10HI) = kcii.k10HI;
    h(i, ThermoTable::ICII_k10H2) = kcii.k10H2;
    h(i, ThermoTable::ICII_boltz) = std::exp(-E10CII / kbT);

    // ----- CI -----
    h(i, ThermoTable::ICI_k10e) = kci.k10e;
    h(i, ThermoTable::ICI_k20e) = kci.k20e;
    h(i, ThermoTable::ICI_k21e) = kci.k21e;
    h(i, ThermoTable::ICI_k10HI) = kci.k10HI;
    h(i, ThermoTable::ICI_k20HI) = kci.k20HI;
    h(i, ThermoTable::ICI_k21HI) = kci.k21HI;
    h(i, ThermoTable::ICI_k10H2) = kci.k10H2;
    h(i, ThermoTable::ICI_k20H2) = kci.k20H2;
    h(i, ThermoTable::ICI_k21H2) = kci.k21H2;
    h(i, ThermoTable::ICI_boltz10) = std::exp(-E10CI / kbT);
    h(i, ThermoTable::ICI_boltz20) = std::exp(-E20CI / kbT);
    h(i, ThermoTable::ICI_boltz21) = std::exp(-E21CI / kbT);

    // ----- OI -----
    h(i, ThermoTable::IOI_k10HI) = koi.k10HI;
    h(i, ThermoTable::IOI_k20HI) = koi.k20HI;
    h(i, ThermoTable::IOI_k21HI) = koi.k21HI;
    h(i, ThermoTable::IOI_k10H2) = koi.k10H2;
    h(i, ThermoTable::IOI_k20H2) = koi.k20H2;
    h(i, ThermoTable::IOI_k21H2) = koi.k21H2;
    h(i, ThermoTable::IOI_k10e) = koi.k10e;
    h(i, ThermoTable::IOI_k20e) = koi.k20e;
    h(i, ThermoTable::IOI_k21e) = koi.k21e;
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

    // ----- H2 rovibrational line cooling -----
    const Thermo::H2CoolCoefs kh2 = Thermo::H2CoolRates(T);
    h(i, ThermoTable::IH2C_LHI) = kh2.LHI;
    h(i, ThermoTable::IH2C_LH2) = kh2.LH2;
    h(i, ThermoTable::IH2C_LHe) = kh2.LHe;
    h(i, ThermoTable::IH2C_LHplus) = kh2.LHplus;
    h(i, ThermoTable::IH2C_Le) = kh2.Le;
    h(i, ThermoTable::IH2C_LTE) = kh2.LTE;

    h(i, ThermoTable::ITEMP) = T;
  }

  Kokkos::deep_copy(tab.data, h);
}

}  // namespace chemistry
