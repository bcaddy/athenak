#ifndef CHEMISTRY_NETWORK_NCR_HPP_
#define CHEMISTRY_NETWORK_NCR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ncr.hpp
//  \brief The implementation of the NCR photochemistry network, a port of the
//  reduced H/C/O network of TIGRESS-NCR (Kim et al. 2023, ApJS 264, 10;
//  tigress-ncr src/photchem/ncr_rates.hpp and ncr_solver.hpp)
//
//  Only Tigress's default code paths are ported: iPEheating = 1 (WD01),
//  iH2heating = 1 (Visser+18), iCII_rec_rate = 2 (Badnell), H2
//  rovibrational cooling of Moseley+21, Smith21 HI cooling off, Chem,
//  PhotDiss, PhotIon and CRPhotC on, x_*_cut = 0, cool_hyd_cie off, no dust
//  thermal balance and no hi_phot_bg. The switches themselves are not ported.
//  The CO LVG cooling uses a constant velocity gradient, <chemistry>/NCR_dvdr,
//  as Tigress's <photchem_ncr>/dvdr, not the gradient of the velocity field.
//
//  Known, documented deviations from Tigress:
//    - x_e is not lagged. Tigress updates x_e after each chemistry substep
//      from the previous x_e. Here x_e = F(x_e, T) is solved jointly with T
//      inside every evaluation of the right hand side, so that it is a pure
//      function of the ODE state. Transients differ, equilibria do not.
//    - H2 above temp_hot1 decays (its formation is off, its destruction is
//      kept) instead of being set to zero.
//    - Constant radiation gives chi_h2 = chi_lw. Tigress applies
//      ShieldingFactorH2(0) = 0.99994, a 6e-5 effect.
//    - Six-ray radiation is a new adaptation: the shared Shielding::fShield_H2
//      and Shielding::fShield_C with NC = 0, i.e. only H2 cross-shielding of
//      CI, and CO dissociation by the dust-attenuated LW band only.
//    - The thresholds and abundances are unified to the inputs: Tigress's
//      constexpr ktemp_hot0/1 (HII grain recombination and the x_e branch)
//      become NCR_temp_hot0/1, and its constexpr kxCstd/kxOstd become
//      NCR_xC*Z_g and NCR_xO*Z_g, as Tigress's solver already uses.
//    - Dropped: the ART, LyC and xi_ph paths (xi_ph_HI = xi_ph_H2 = 0), the
//      local shielding length and local CR attenuation, the dust thermal
//      balance, the option switches, and the timestep feedback.
//    - Small-value floors and guards are 1e-20 (Tigress's TINY_NUMBER)
//      instead of Tigress's small_ = 1e-50 (GetxCII, the HII and C+ grain
//      recombination psi, and the H2 collisional dissociation n_crit). Floors
//      applied before a log in the rate tables are precision-aware,
//      1000*std::numeric_limits<Real>::min() as Tigress's SetLogData, instead
//      of Tigress's kRateFloor = 1e-100; this changes the tabulated k_coll,HI
//      only in the grid interval that straddles its 3000 K cutoff.
//    - CONcritFactor_ at xi_cr = 0 uses the limit of the CO critical density
//      instead of Tigress's inf * 0 = NaN.
//  Smaller differences, from reusing AthenaK code where it is otherwise
//  identical to Tigress:
//    - Thermo uses eV = 1.602e-12 erg, Tigress 1.60218e-12 erg. The reused
//      terms proportional to Thermo's eV (CR heating, the three H2 heating
//      terms and HI collisional ionization cooling) are multiplied by
//      kEvRatio_. HeatingPE and CoolingRec don't use eV.
//    - k_B is AthenaK's 1.3806488e-16, Tigress's is 1.38065e-16.
//    - Thermo::HeatingPE returns 0 for n_e < 1e-10 cm^-3.
//    - rates::CIIRecombination computes T^-1.5 from sqrt(T/T0) rather than
//      with pow, and several powers here are written as exp(e*lnT) with a
//      shared lnT. Both are algebraically identical to Tigress.
//    - Every rate sees T >= kTempFloorRates_ = 1 K, and the T/mu floor is
//      applied by switching off net cooling (Edot = 0) rather than by flooring
//      the pressure.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#include "athena.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/thermo/ncr_cie_table.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
struct NCRSettings {
  /// Gas metallicity relative to solar, scales the C and O abundances
  Real zg;
  /// Dust metallicity relative to solar
  Real zd;
  /// Total C abundance per H, <chemistry>/NCR_xC times zg
  Real xC_tot;
  /// Total O abundance per H, <chemistry>/NCR_xO times zg
  Real xO_tot;
  /// Temperature in K above which the CIE table takes over: x_e from the
  /// table, and the non-equilibrium cooling and heating are blended out
  Real temp_hot0;
  /// Temperature in K above which there is only CIE cooling, no H2 formation
  /// and no grain recombination
  Real temp_hot1;
  /// Floor on T/mu in K. Net cooling is switched off at or below it
  Real temp_mu_floor;
  /// Rate interpolation: 0 exact, 1 log-log, 2 NQT2 and 3 NQT1 tables
  int interp_mode;
  /// Hold the temperature fixed and evolve only the chemistry
  bool chem_only;

  /// Velocity gradient in s^-1 for the CO LVG cooling, Tigress's dvdr
  Real dvdr;

  // ----- Rate tables, network-internal, empty in exact mode -----
  // Managed Views, so every copy of the settings shares the tables and the
  // last copy to go frees them
  /// The full-range table, NCRNetwork::kNTabT_ by NCRNetwork::kNTabFull_
  DvceArray2D<Real> tab_full;
  /// The cold-range table, NCRNetwork::kNTabTCold_ by NCRNetwork::kNTabCold_
  DvceArray2D<Real> tab_cold;
  /// NQT modes only: nqt1_log(T) of the first row of both tables
  double nqt_t_min;
  /// NQT modes only: rows per unit nqt1_log(T) of the full-range table
  double nqt_idt;
  /// NQT modes only: rows per unit nqt1_log(T) of the cold-range table
  double nqt_idt_cold;
};

/*!
 * \brief The class for the NCR network of TIGRESS-NCR (Kim et al. 2023).
 *
 * \details The ODE variables are x_H2, x_H+ and the internal energy, as for
 * every network, see GOW17Network. x_HI follows from conservation. C+, CO,
 * CI, OI and O+ are algebraic, in equilibrium with the evolved species, and
 * the electron abundance x_e is solved together with the temperature inside
 * every evaluation of the right hand side, see SolveElectrons_.
 *
 * Radiation bands. The bands are chosen so that the existing radiation
 * functors give NCR's fields without change:
 *   - iph_LW: dust-attenuated LW, Tigress's chi_lw. Also chi_co, which is
 *     dust-attenuated only (Kim+23 section 3.2.1).
 *   - iph_H2: LW times the H2 self-shielding factor, chi_h2.
 *   - iph_C: LW times the CI shielding factor, chi_ci.
 *   - All three LW bands use kph_avfac = 3.74, i.e. sigma_d,LW = 2e-21 cm^2
 *     times 1.87e21 cm^-2.
 *   - irad_GPE: chi_pe. Its dust cross section, 1e-21 cm^2, is NCR's PE one.
 *   - irad_CR: the cosmic ray ionization rate xi_CR.
 *   - chi_fuv is the ISRF-weighted mean of chi_pe and chi_lw.
 *   - NCR has no CO band. iph_CO = -1 is a sentinel that no band index ever
 *     matches, so ShieldedRadiationFunc applies no CO shielding.
 */
class NCRNetwork {
 public:
  /*!
   * \tparam RadFunc_t The radiation functor, Radiation::ConstantRadiationFunc
   * or Radiation::ShieldedRadiationFunc, called once here for the cell's
   * radiation field
   */
  template <typename RadFunc_t>
  KOKKOS_FUNCTION NCRNetwork(NCRSettings const settings, const int mb_idx,
                             const int k, const int j, const int i,
                             const DvceArray5D<Real> w0,
                             const DualArray1D<RegionSize> sizes,
                             const RadFunc_t& rad_func,
                             Real const density_cgs, Real const mu_H,
                             Real const gamma, Real const hydrogen_mass_cgs,
                             Real const units_time_cgs,
                             Real const units_energy_density_cgs)
      : n_H(w0(mb_idx, IDN, k, j, i) * density_cgs /
            (mu_H * hydrogen_mass_cgs)),
        gamma(gamma),
        units_time_cgs(units_time_cgs),
        units_energy_density_cgs(units_energy_density_cgs),
        y(y_buffer_, neqs),
        y_new(y_new_buffer_, neqs),
        settings_(settings),
        mu_H_(mu_H),
        rad_(MakeRadFields_(rad_func(mb_idx, k, j, i))),
        co_ncrit_fac_(CONcritFactor_(settings, rad_, n_H)),
        T0_(0.0) {}

  // ----- Number of equations -----
  static constexpr int neqs = 3;

  // ----- Species indices within the ODE system ------
  // The order must be: real species then internal energy
  enum : size_t {
    IH2,      // H2
    IH_plus,  // H+
    IIE,      // internal energy, must be last
  };

  // ----- Names, used for output, must be the same order as the enum -----
  static constexpr std::array<std::string_view, neqs - 1> species_names = {
      "H2", "H+"};

  /// He abundance per H, Tigress's x_he_
  static constexpr Real xHe = 0.1;

  // ----- Radiation Constants -----
  static constexpr int n_ph = 3;
  static constexpr int n_freq = n_ph + 2;
  /// Index of each LW band in the radiation field, see the class comment
  enum : size_t { iph_LW, iph_H2, iph_C };
  /// Index of the photoelectric heating and cosmic ray bands in the radiation
  /// field
  enum : size_t { irad_GPE = n_ph, irad_CR };
  /// Sentinel: NCR has no CO band. No band index equals it, so the radiation
  /// functors apply no CO self-shielding
  static constexpr int iph_CO = -1;
  /// Exponent factor in front of AV for the dust shielding of each LW band,
  /// sigma_d,LW = 2e-21 cm^2 times 1.87e21 cm^-2
  static constexpr Kokkos::Array<Real, n_ph> kph_avfac = {3.74, 3.74, 3.74};

  // ----- cell values -----
  Real const n_H;  // The number density of hydrogen
  Real const gamma;

  // ----- unit conversion factors -----
  Real const units_time_cgs;
  Real const units_energy_density_cgs;

  // ----- Views to store ODE state -----
  // The current state
  Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> y;
  // The results of evaluating the ODEs
  Kokkos::View<Real*, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> y_new;

  // ----- Member Functions -----
  /*!
   * \brief Get the settings for the NCR network from the input file, and
   * build the rate tables in a table interpolation mode. Every call builds
   * its own tables, which its settings own
   *
   * \param pin The ParameterInput object
   * \param ppack The MeshBlockPack, unused
   * \return NCRSettings The settings for the NCR network
   */
  static NCRSettings GetSettings(ParameterInput* pin, MeshBlockPack* ppack) {
    NCRSettings output;

    // Metallicities
    output.zg = pin->GetOrAddReal("chemistry", "NCR_Z_g", 1.0);
    output.zd = pin->GetOrAddReal("chemistry", "NCR_Z_d", 1.0);
    // C and O abundances, input at Z_g = 1. These replace Tigress's
    // constexpr kxCstd and kxOstd as well, so that the rates and the solver
    // agree
    output.xC_tot =
        pin->GetOrAddReal("chemistry", "NCR_xC", 1.6e-4) * output.zg;
    output.xO_tot =
        pin->GetOrAddReal("chemistry", "NCR_xO", 3.2e-4) * output.zg;

    // Temperature thresholds. These replace Tigress's constexpr ktemp_hot0/1
    // as well, so that every threshold follows the input
    output.temp_hot0 = pin->GetOrAddReal("chemistry", "NCR_temp_hot0", 2.0e4);
    output.temp_hot1 = pin->GetOrAddReal("chemistry", "NCR_temp_hot1", 3.5e4);
    output.temp_mu_floor =
        pin->GetOrAddReal("chemistry", "NCR_temp_mu_floor", 2.0);

    output.interp_mode =
        pin->GetOrAddInteger("chemistry", "NCR_interp_mode", kInterpLogLog);
    output.chem_only =
        pin->GetOrAddBoolean("chemistry", "NCR_chem_only", false);

    // Constant velocity gradient for the CO LVG cooling, Tigress's default
    // 1 km/s/pc
    output.dvdr =
        pin->GetOrAddReal("chemistry", "NCR_dvdr", 3.240779289444365e-14);

    // Above temp_hot0 x_e comes from the CIE table, which starts at 1e4 K
    const Real temp_table_min = std::pow(10.0, NCRCIETable::log_T_min);
    if (output.temp_hot0 < temp_table_min) {
      std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<chemistry>/NCR_temp_hot0 = " << output.temp_hot0
                << " K is below the lower limit " << temp_table_min
                << " K of the NCR CIE table" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!(output.temp_hot1 > output.temp_hot0)) {
      std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<chemistry>/NCR_temp_hot1 = " << output.temp_hot1
                << " must be greater than <chemistry>/NCR_temp_hot0 = "
                << output.temp_hot0 << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (output.interp_mode < kInterpExact || output.interp_mode > kInterpNqt1) {
      std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<chemistry>/NCR_interp_mode = " << output.interp_mode
                << " is not a valid option, choose from {0 (exact), "
                   "1 (log-log), 2 (NQT2), 3 (NQT1)}"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // NQT grid parameters, Tigress's nqt_t_min_, nqt_idt_ and nqt_idt_cold_.
    // kNqt2 shares the nqt1_log(T) grid of kNqt1
    output.nqt_t_min = 0.0;
    output.nqt_idt = 0.0;
    output.nqt_idt_cold = 0.0;
    if (output.interp_mode == kInterpNqt2 ||
        output.interp_mode == kInterpNqt1) {
      output.nqt_t_min = Nqt1Log_(std::pow(10.0, kLogTMin_));
      output.nqt_idt = (kNTabT_ - 1) / (Nqt1Log_(std::pow(10.0, kLogTMax_)) -
                                        output.nqt_t_min);
      output.nqt_idt_cold =
          (kNTabTCold_ - 1) /
          (Nqt1Log_(std::pow(10.0, kLogTMaxCold_)) - output.nqt_t_min);
    }

    // Rate tables
    if (output.interp_mode != kInterpExact) {
      output.tab_full =
          DvceArray2D<Real>("NCR_rate_table_full", kNTabT_, kNTabFull_);
      output.tab_cold =
          DvceArray2D<Real>("NCR_rate_table_cold", kNTabTCold_, kNTabCold_);
      BuildRateTables(output);
    }

    return output;
  }

  /*!
   * \brief Fill the rate tables on the device (Tigress's BuildTables) by
   * evaluating the exact rate functions, so that every mode shares one
   * formula. Public since nvcc requires the function enclosing a
   * KOKKOS_LAMBDA to be.
   *
   * \details The log-log tables are on a uniform log10(T) grid and store
   * log10(rate), the NQT tables are on a uniform nqt1_log(T) grid and store
   * nqt2_log(rate) or nqt1_log(rate). Rates are floored at kRateFloor_ before
   * the log, for those that are exactly zero below a cutoff.
   *
   * \param s The settings, with interp_mode 1, 2 or 3, the NQT grid, and the
   * allocated tables to fill
   */
  static void BuildRateTables(const NCRSettings& s) {
    const bool is_log_log = (s.interp_mode == kInterpLogLog);

    par_for(
        "NCR_rate_table_full", DevExeSpace(), 0, kNTabT_ - 1,
        KOKKOS_LAMBDA(const int i) {
          const Real T = is_log_log
                             ? Kokkos::pow(10.0, kLogTMin_ + i * kDLogT_)
                             : static_cast<Real>(
                                   Nqt1Exp_(s.nqt_t_min + i / s.nqt_idt));
          const Real lnT = Kokkos::log(T);
          s.tab_full(i, kTabAlphaBHII_) =
              EncodeRate_(s.interp_mode, AlphaBHII(lnT));
          s.tab_full(i, kTabCollIonHI_) =
              EncodeRate_(s.interp_mode, CollIonHI(T, lnT));
        });

    par_for(
        "NCR_rate_table_cold", DevExeSpace(), 0, kNTabTCold_ - 1,
        KOKKOS_LAMBDA(const int i) {
          const Real T = is_log_log
                             ? Kokkos::pow(10.0, kLogTMin_ + i * kDLogTCold_)
                             : static_cast<Real>(
                                   Nqt1Exp_(s.nqt_t_min + i / s.nqt_idt_cold));
          const Real lnT = Kokkos::log(T);
          s.tab_cold(i, kTabRecombCII_) =
              EncodeRate_(s.interp_mode, rates::CIIRecombination(T));
          s.tab_cold(i, kTabKgrH2_) =
              EncodeRate_(s.interp_mode, rates::H2FormationGrainHM79(T));
          // Each transition is stored as e, HI, H2, and each 3 level atom as
          // k10, k20, k21, as ColdColliderRates_ and ColdThreeLevelRates_
          // read them
          auto store = [&](const int col, const ColliderRates& k) {
            s.tab_cold(i, col) = EncodeRate_(s.interp_mode, k.e);
            s.tab_cold(i, col + 1) = EncodeRate_(s.interp_mode, k.HI);
            s.tab_cold(i, col + 2) = EncodeRate_(s.interp_mode, k.H2);
          };
          auto store3 = [&](const int col10, const ThreeLevelRates& k) {
            store(col10, k.k10);
            store(col10 + 3, k.k20);
            store(col10 + 6, k.k21);
          };
          const Real lnT2 = lnT - Thermo::ln100_;
          store(kTabK10CII_, Thermo::RatesCII(T, lnT2));
          store3(kTabOI_, OIRates(lnT));
          store3(kTabCI_, Thermo::RatesCI(T, lnT, lnT2));
        });
    Kokkos::fence();
  }

  /*!
   * \brief The electron abundance at a FIXED temperature: the bracketed,
   * safeguarded solution of x_e = F(x_e, T), see SolveElectrons_. Used by
   * the problem generator and by the chem_only hooks. The abundances are
   * floored as in evaluate_function. In a table interpolation mode the
   * tables are in device memory, so call this in device code, e.g. a par_for.
   *
   * \param settings The network settings
   * \param rad The radiation field in each band, as from the radiation
   * functors
   * \param n_H The number density of H nuclei in cm^-3
   * \param T The temperature in K
   * \param xH2 The H2 abundance per H
   * \param xH_plus The H+ abundance per H
   * \return Real The electron abundance per H
   */
  KOKKOS_FUNCTION static Real ElectronAbundance(
      const NCRSettings& settings, const RegisterArray<Real, n_freq>& rad,
      const Real n_H, const Real T, const Real xH2, const Real xH_plus) {
    return ElectronsAtT_(settings, MakeRadFields_(rad), n_H, FloorH2_(xH2),
                         FloorHPlus_(xH_plus), T);
  }

  /*!
   * \brief chem_only hook, run once y is loaded: pin the temperature T0_ to
   * the one of the loaded internal energy, with x_e solved jointly. A no-op
   * unless chem_only
   */
  KOKKOS_FUNCTION void PreSolve() {
    if (!settings_.chem_only) {
      return;
    }
    const Real E_ergs = y(IIE) * units_energy_density_cgs / n_H;
    T0_ = SolveElectrons_(settings_, rad_, n_H, FloorH2_(y(IH2)),
                          FloorHPlus_(y(IH_plus)), E_ergs, gamma, false, 0.0)
              .T;
  }

  /*!
   * \brief chem_only hook, run before y is written back: set the internal
   * energy so that the temperature stays T0_ with the new abundances, as
   * Tigress's chem_only UpdateTemperature does with the pressure. A no-op
   * unless chem_only
   */
  KOKKOS_FUNCTION void PostSolve() {
    if (!settings_.chem_only) {
      return;
    }
    const Real xH2 = FloorH2_(y(IH2));
    const Real xe =
        ElectronsAtT_(settings_, rad_, n_H, xH2, FloorHPlus_(y(IH_plus)), T0_);
    y(IIE) = n_H * Thermo::CvCold(xH2, xHe, xe, gamma) * T0_ /
             units_energy_density_cgs;
  }

  /*!
   * \brief Computes `f` using the values in `y_in`
   *
   * \details The rate coefficients see a floored copy of the state: x_H2 and
   * x_H+ floored at 1e-20, as Tigress's state always is, and x_HI =
   * max(1 - x_H+ - 2 x_H2, 1e-20). The linear source terms use the unfloored
   * x_HI = 1 - x_H+ - 2 x_H2 instead, in GOW17's `creation - y_in *
   * destruction` form, so that f is smooth for the numerical Jacobian even
   * near full ionization. y_in is never changed.
   */
  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION __attribute__((noinline)) void evaluate_function(
      const Real /*t*/, const Real /*dt*/, const vec_type1& y_in,
      vec_type2& f) const {
    for (int i = 0; i < neqs; i++) {
      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(y_in(i)) || Kokkos::isnan(y_in(i))) {
        Kokkos::abort("Error: NaN or Inf value found in NCR `y` array\n");
      }
    }

    // ----- Local state -----
    // Floored abundances, for the rate coefficients only
    const Real xH2 = FloorH2_(y_in(IH2));
    const Real xH_plus = FloorHPlus_(y_in(IH_plus));
    const Real xHI = Kokkos::fmax(1.0 - xH_plus - 2.0 * xH2, kTiny_);
    // Unfloored HI, for the linear source terms
    const Real xHI_src = 1.0 - y_in(IH_plus) - 2.0 * y_in(IH2);

    // ----- Electrons and temperature -----
    // T is fixed at T0_ in chem_only, else set by the internal energy
    const Real E_ergs = y_in(IIE) * units_energy_density_cgs / n_H;
    const ElectronSolution es =
        SolveElectrons_(settings_, rad_, n_H, xH2, xH_plus, E_ergs, gamma,
                        settings_.chem_only, T0_);
    const Real T = es.T;
    const Real lnT = es.lnT;
    const Real ne = es.xe * n_H;
    // Weight of the CIE side of the sigmoid, used only in [temp_hot0,
    // temp_hot1)
    const Real w_hot =
        (T >= settings_.temp_hot0 && T < settings_.temp_hot1)
            ? HotWeight_(settings_, T)
            : 0.0;

    // ----- Rate coefficients, Tigress's HIIRates and H2Rates -----
    const Real alpha_B = AlphaB_(settings_, T, lnT);
    const Real k_coll_HI = CollIonHI_(settings_, T, lnT);
    const Real xi_coll_H2 = CollDissH2_(n_H, T, lnT, xHI, xH2);
    // H2 formation on grains, off above temp_hot1
    const Real kgr =
        (T < settings_.temp_hot1) ? KgrH2_(settings_, T, lnT) : 0.0;

    // H+: collisional and cosmic ray ionization (xi_ph_HI = 0), radiative
    // and grain recombination
    const Real xi_cr_HI = rad_.xi_cr * (2.3 * xH2 + 1.5 * xHI);
    const Real C_HII = k_coll_HI * ne + xi_cr_HI;
    const Real D_HII =
        ne * alpha_B +
        n_H * GrainRecombHII_(settings_, rad_, ne, T, lnT, w_hot);

    // H2: formation on grains, destruction by cosmic rays, photodissociation
    // (xi_ph_H2 = 0) and collisional dissociation
    const Real xi_cr_H2 = 2.0 * xi_cr_HI;
    const Real C_H2 = kgr * n_H;
    const Real D_H2 = 1.65 * xi_cr_H2 + rad_.xi_diss_h2 + xi_coll_H2;

    // ----- Species -----
    f(IH2) = (C_H2 * xHI_src - y_in(IH2) * D_H2) * units_time_cgs;
    f(IH_plus) = (C_HII * xHI_src - y_in(IH_plus) * D_HII) * units_time_cgs;

    // ----- Internal energy equation -----
    if (settings_.chem_only) {
      f(IIE) = 0.0;
    } else {
      f(IIE) = Edot_(es, xHI, xH2, xH_plus, alpha_B, k_coll_HI, kgr,
                     xi_coll_H2, w_hot);
    }

    // Verify the results are finite and not NaN valued
    for (int i = 0; i < neqs; i++) {
      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(f(i)) || Kokkos::isnan(f(i))) {
        Kokkos::abort("Error: NaN or Inf value found in NCR `f` array\n");
      }
    }
  }

  template <class vec_type, class mat_type>
  KOKKOS_FUNCTION void evaluate_jacobian(const Real t, const Real dt,
                                         const vec_type& y_in,
                                         const mat_type& jac) const {
    chemistry::numerical_jacobian(*this, t, dt, y_in, jac);
  }

 private:
  // ----- Raw array buffers -----
  // The current state
  Real y_buffer_[neqs];  // NOLINT(runtime/arrays)
  // The results of evaluating the ODEs
  Real y_new_buffer_[neqs];  // NOLINT(runtime/arrays)

  /// The cell's radiation fields, Tigress's NCRVar members of the same names
  struct RadFields {
    Real chi_fuv;     ///< ISRF-weighted mean of chi_pe and chi_lw
    Real chi_ci;      ///< LW field that ionizes CI
    Real chi_co;      ///< LW field that dissociates CO
    Real xi_cr;       ///< cosmic ray ionization rate per H in s^-1
    Real xi_diss_h2;  ///< H2 photodissociation rate per H2 in s^-1
  };

  /// A solution of x_e = F(x_e, T)
  struct ElectronSolution {
    Real xe;     ///< electron abundance per H
    Real T_raw;  ///< temperature in K, before the rate floor
    Real T;      ///< temperature in K seen by the rates, >= kTempFloorRates_
    Real lnT;    ///< ln(T)
    /// The equilibrium C+ abundance per H at xe and T, if has_xCII
    Real xCII;
    /// Whether xCII is set, i.e. the cold branch last evaluated xe itself
    bool has_xCII;
  };

  /// A temperature's row and fractional position on a rate table's grid
  struct TablePoint {
    int i;
    Real frac;
  };

  using ColliderRates = Thermo::ColliderRates;
  using ThreeLevelRates = Thermo::ThreeLevelRates;

  // ----- Settings and cell values -----
  const NCRSettings settings_;
  /// Mean mass per H in units of the H mass, <chemistry>/mu_H
  const Real mu_H_;
  /// The cell's radiation fields, from the radiation functor
  const RadFields rad_;
  /// The CO abundance factor 1 / (1 + n_crit^2 / n_H^2), which depends only
  /// on the cell's density and radiation, see CONcritFactor_
  const Real co_ncrit_fac_;
  /// chem_only: the pinned temperature in K, set by PreSolve
  Real T0_;

  // ----- Constants -----
  /// Small-value floor and guard, Tigress's TINY_NUMBER
  static constexpr Real kTiny_ = 1.0e-20;
  /// Every rate sees a temperature of at least this, in K
  static constexpr Real kTempFloorRates_ = 1.0;
  /// The x_e solve converges when an iteration moves x_e by at most this
  /// fraction. Tight enough that f is smooth at the 1.5e-8 perturbations of
  /// the numerical Jacobian
  static constexpr Real kXeRelTol_ = 1.0e-14;
  /// Iteration cap of the x_e solve, after which it returns the bracket
  /// midpoint
  static constexpr int kXeMaxIter_ = 100;
  /// Cap on the widenings of an x_e bracket
  static constexpr int kXeMaxWiden_ = 10;
  /// Relative margin above temp_hot0 of T(hi) for skipping the cold x_e
  /// solve, see SolveElectrons_. It covers the rounding by which the cold
  /// root can exceed hi, so that the skip never changes the result
  static constexpr Real kHotSkipMargin_ =
      1.0e4 * std::numeric_limits<Real>::epsilon();
  /// Heating floor in erg s^-1 H^-1 above temp_hot1
  static constexpr Real kHeatMin_ = 1.0e-35;
  /// HI cooling and collisional ionization only above this, in K
  static constexpr Real kTempHI_ = 3.0e3;
  /// H2 collisional dissociation only above this, in K
  static constexpr Real kTempCollDissH2_ = 7.0e2;
  /// eV in erg, Tigress's Constants::electron_volt_cgs
  static constexpr Real kEv_ = 1.60218e-12;
  /// Corrects terms computed with Thermo's eV_ = 1.602e-12 erg to kEv_
  static constexpr Real kEvRatio_ = kEv_ / Thermo::eV_;
  /// Maximum effective length for the CO LVG cooling in cm, 100 pc
  static constexpr Real kLeffCO_ = 3.086e20;
  /// ISRF energy densities in the PE and LW bands in erg cm^-3
  static constexpr Real kURadPEIsrf_ = 7.613e-14;
  static constexpr Real kURadLWIsrf_ = 1.335e-14;
  /// H2 photodissociation rate in the ISRF in s^-1
  static constexpr Real kXiDissH2Isrf_ = 5.7e-11;
  /// Maximum free electron abundance per H, in GetxeMol_
  static constexpr Real kxELMax_ = 1.2006199779862501;
  /// Floor on tabulated rates before their log, Tigress's SetLogData floor
  static constexpr Real kRateFloor_ = 1000 * std::numeric_limits<Real>::min();

  // Natural logs, so that powers and logs of scaled temperatures reuse lnT.
  // Thermo::ln10_, ln100_ and ln5_ are also used
  static constexpr Real kLog10e_ = 4.34294481903251761e-1;  // 1/ln(10)
  static constexpr Real kLn1e4_ = 9.21034037197618360e0;
  static constexpr Real kLn8_6173em5_ = -9.35915365441784530e0;
  static constexpr Real kLn3_9810717em6_ = -1.24339595035581691e1;

  // ----- Rate tables, Tigress's grids -----
  enum : int { kInterpExact, kInterpLogLog, kInterpNqt2, kInterpNqt1 };
  /// Full-range grid, alpha_B,HII and k_coll,HI, active at all T
  static constexpr Real kLogTMin_ = 1.0;  // log10(10 K)
  static constexpr Real kLogTMax_ = 8.5;  // log10(~3e8 K)
  static constexpr int kNTabT_ = 2000;
  static constexpr Real kDLogT_ = (kLogTMax_ - kLogTMin_) / (kNTabT_ - 1);
  static constexpr Real kIDLogT_ = 1.0 / kDLogT_;
  /// Cold-range grid, the rates only used below temp_hot0/1
  static constexpr Real kLogTMaxCold_ = 5.0;  // log10(1e5 K)
  static constexpr int kNTabTCold_ = 1000;
  static constexpr Real kDLogTCold_ =
      (kLogTMaxCold_ - kLogTMin_) / (kNTabTCold_ - 1);
  static constexpr Real kIDLogTCold_ = 1.0 / kDLogTCold_;
  /// Columns of the full-range table, Tigress's TabFull
  enum : int { kTabAlphaBHII_, kTabCollIonHI_, kNTabFull_ };
  /// Columns of the cold-range table, Tigress's TabCold. Each transition is
  /// stored as e, HI, H2, and each 3 level atom as k10, k20, k21, see
  /// ColdThreeLevelRates_
  enum : int {
    kTabRecombCII_ = 0,
    kTabKgrH2_ = 1,
    kTabK10CII_ = 2,
    kTabOI_ = 5,
    kTabCI_ = 14,
    kNTabCold_ = 23
  };
  // NQT bit constants: as_int(1.0) and as_int(2.0) - as_int(1.0) = 2^52
  static constexpr std::int64_t kNqtOne_ = 4607182418800017408LL;
  static constexpr double kNqtMantissa_ = 4503599627370496.0;
  static constexpr std::int64_t kNqtMantissaMask_ = 0x000FFFFFFFFFFFFFLL;

  //-------------------------------------------------------------------------------------
  // NQT helpers, Hammond et al. 2025, ApJS 277:65. In double whatever Real is
  /// NQTo1: piecewise-linear approximant of log2(x) from the IEEE 754 bits
  KOKKOS_INLINE_FUNCTION static double Nqt1Log_(const double x) {
    const std::int64_t i = Kokkos::bit_cast<std::int64_t>(x);
    return static_cast<double>(i - kNqtOne_) / kNqtMantissa_;
  }
  /// Inverse of Nqt1Log_
  KOKKOS_INLINE_FUNCTION static double Nqt1Exp_(const double y) {
    const std::int64_t i =
        static_cast<std::int64_t>(y * kNqtMantissa_) + kNqtOne_;
    return Kokkos::bit_cast<double>(i);
  }
  /// NQTo2: Nqt1Log_ plus the quadratic correction b*(1-b)/3, b the
  /// fractional mantissa
  KOKKOS_INLINE_FUNCTION static double Nqt2Log_(const double x) {
    const std::int64_t i = Kokkos::bit_cast<std::int64_t>(x);
    const double b =
        static_cast<double>(i & kNqtMantissaMask_) / kNqtMantissa_;
    return static_cast<double>(i - kNqtOne_) / kNqtMantissa_ +
           b * (1.0 - b) / 3.0;
  }
  /// Inverse of Nqt2Log_, b = 2 - sqrt(4 - 3f). Needs the true floor, since
  /// floored rates give large negative y
  KOKKOS_INLINE_FUNCTION static double Nqt2Exp_(const double y) {
    const std::int64_t e = static_cast<std::int64_t>(Kokkos::floor(y));
    const double f = y - static_cast<double>(e);
    const double b = 2.0 - Kokkos::sqrt(4.0 - 3.0 * f);
    const std::int64_t mantissa =
        static_cast<std::int64_t>(b * kNqtMantissa_);
    const std::int64_t i = ((e + 1023) << 52) | mantissa;
    return Kokkos::bit_cast<double>(i);
  }

  //-------------------------------------------------------------------------------------
  // Table encoding and lookup
  /// A rate as stored in the table of interp_mode
  KOKKOS_INLINE_FUNCTION static Real EncodeRate_(const int interp_mode,
                                                 const Real rate) {
    const Real v = Kokkos::fmax(rate, kRateFloor_);
    if (interp_mode == kInterpLogLog) {
      return Kokkos::log10(v);
    } else if (interp_mode == kInterpNqt2) {
      return static_cast<Real>(Nqt2Log_(v));
    }
    return static_cast<Real>(Nqt1Log_(v));
  }

  /*!
   * \brief The row and fraction of T on a table's grid. As Tigress, the row
   * is clamped to [0, n_tab - 2] but the fraction isn't, so lookups outside
   * the grid extrapolate linearly from the end intervals
   */
  KOKKOS_INLINE_FUNCTION static TablePoint GetTablePoint_(const NCRSettings& s,
                                                          const bool is_cold,
                                                          const Real T,
                                                          const Real lnT) {
    Real idx;
    if (s.interp_mode == kInterpLogLog) {
      // log10(T) from the shared lnT
      idx = (lnT * kLog10e_ - kLogTMin_) * (is_cold ? kIDLogTCold_ : kIDLogT_);
    } else {
      idx = (Nqt1Log_(T) - s.nqt_t_min) *
            (is_cold ? s.nqt_idt_cold : s.nqt_idt);
    }
    const int n_tab = is_cold ? kNTabTCold_ : kNTabT_;
    const Real idx_clamped =
        Kokkos::fmax(0.0, Kokkos::fmin(idx, static_cast<Real>(n_tab - 1)));
    const int i = Kokkos::min(static_cast<int>(idx_clamped), n_tab - 2);
    return {i, idx - i};
  }

  /// The rate in column col of tab at p, decoded for the interpolation mode
  KOKKOS_INLINE_FUNCTION static Real TableRate_(
      const NCRSettings& s, const DvceArray2D<Real>& tab, const TablePoint& p,
      const int col) {
    const Real v0 = tab(p.i, col);
    const Real v = v0 + p.frac * (tab(p.i + 1, col) - v0);
    if (s.interp_mode == kInterpLogLog) {
      // pow(10, v)
      return Kokkos::exp(Thermo::ln10_ * v);
    } else if (s.interp_mode == kInterpNqt2) {
      return static_cast<Real>(Nqt2Exp_(v));
    }
    return static_cast<Real>(Nqt1Exp_(v));
  }

  /// The collider rates of one transition, stored at col (e), col+1 (HI)
  /// and col+2 (H2) of the cold-range table
  KOKKOS_INLINE_FUNCTION static ColliderRates ColdColliderRates_(
      const NCRSettings& s, const TablePoint& p, const int col) {
    ColliderRates k;
    k.e = TableRate_(s, s.tab_cold, p, col);
    k.HI = TableRate_(s, s.tab_cold, p, col + 1);
    k.H2 = TableRate_(s, s.tab_cold, p, col + 2);
    return k;
  }

  /// The collider rates of a 3 level atom, stored as k10, k20 and k21 from
  /// col10 of the cold-range table
  KOKKOS_INLINE_FUNCTION static ThreeLevelRates ColdThreeLevelRates_(
      const NCRSettings& s, const TablePoint& p, const int col10) {
    ThreeLevelRates k;
    k.k10 = ColdColliderRates_(s, p, col10);
    k.k20 = ColdColliderRates_(s, p, col10 + 3);
    k.k21 = ColdColliderRates_(s, p, col10 + 6);
    return k;
  }

  //-------------------------------------------------------------------------------------
  // Exact rate coefficients, functions of T only
  /*!
   * \brief Case B radiative recombination of H+, Ferland et al. (1992)
   *
   * \param lnT ln(T), T the temperature in K
   * \return Real The rate coefficient in cm^3 s^-1
   */
  KOKKOS_INLINE_FUNCTION static Real AlphaBHII(const Real lnT) {
    return 2.753e-14 * rates::CaseBRecombHII(lnT);
  }

  /*!
   * \brief Collisional ionization of HI by electrons, Janev et al. (1987),
   * zero at and below 3000 K
   *
   * \param T The temperature in K
   * \param lnT ln(T)
   * \return Real The rate coefficient in cm^3 s^-1
   */
  KOKKOS_INLINE_FUNCTION static Real CollIonHI(const Real T, const Real lnT) {
    if (!(T > kTempHI_)) {
      return 0.0;
    }
    // ln(T*8.6173e-5), i.e. ln of T in eV
    return rates::HICollIonJanev87(lnT + kLn8_6173em5_);
  }

  /*!
   * \brief De-excitation of the OI fine structure lines by e, HI and H2,
   * Tigress's OIRates_. The H2 coefficients are NCR's, which differ from
   * Thermo::CoolingOI's
   *
   * \param lnT ln(T), T the temperature in K
   * \return ThreeLevelRates k10, k20 and k21 in cm^3 s^-1
   */
  KOKKOS_INLINE_FUNCTION static ThreeLevelRates OIRates(const Real lnT) {
    const Real lnT2 = lnT - Thermo::ln100_;
    constexpr Real fo = Thermo::fo_;
    constexpr Real fp = Thermo::fp_;
    ThreeLevelRates k;
    k.k10.HI = 3.57e-10 * Kokkos::exp((0.419 - 0.003 * lnT2) * lnT2);
    k.k20.HI = 3.19e-10 * Kokkos::exp((0.369 - 0.006 * lnT2) * lnT2);
    k.k21.HI = 4.34e-10 * Kokkos::exp((0.755 - 0.160 * lnT2) * lnT2);
    k.k10.H2 = (1.49e-10 * Kokkos::exp((0.369 - 0.026 * lnT2) * lnT2)) * fp +
               (1.37e-10 * Kokkos::exp((0.395 - 0.005 * lnT2) * lnT2)) * fo;
    k.k20.H2 = (2.37e-10 * Kokkos::exp((0.255 + 0.016 * lnT2) * lnT2)) * fp +
               (2.23e-10 * Kokkos::exp((0.284 + 0.035 * lnT2) * lnT2)) * fo;
    k.k21.H2 = (2.10e-12 * Kokkos::exp((1.117 + 0.070 * lnT2) * lnT2)) * fp +
               (3.00e-12 * Kokkos::exp((0.792 + 0.188 * lnT2) * lnT2)) * fo;
    k.k10.e = 5.12e-10 * Kokkos::exp(-0.075 * lnT);
    k.k20.e = 4.86e-10 * Kokkos::exp(-0.026 * lnT);
    k.k21.e = 1.08e-14 * Kokkos::exp(0.926 * lnT);
    return k;
  }

  //-------------------------------------------------------------------------------------
  // Rate coefficients in the interpolation mode of the settings
  /// Case B recombination of H+ in cm^3 s^-1
  KOKKOS_INLINE_FUNCTION static Real AlphaB_(const NCRSettings& s,
                                             const Real T, const Real lnT) {
    if (s.interp_mode == kInterpExact) {
      return AlphaBHII(lnT);
    }
    return TableRate_(s, s.tab_full, GetTablePoint_(s, false, T, lnT),
                      kTabAlphaBHII_);
  }

  /// Collisional ionization of HI in cm^3 s^-1
  KOKKOS_INLINE_FUNCTION static Real CollIonHI_(const NCRSettings& s,
                                                const Real T, const Real lnT) {
    if (s.interp_mode == kInterpExact) {
      return CollIonHI(T, lnT);
    }
    return TableRate_(s, s.tab_full, GetTablePoint_(s, false, T, lnT),
                      kTabCollIonHI_);
  }

  /// Radiative and dielectronic recombination of C+ in cm^3 s^-1
  KOKKOS_INLINE_FUNCTION static Real RecombCII_(const NCRSettings& s,
                                                const Real T, const Real lnT) {
    if (s.interp_mode == kInterpExact) {
      return rates::CIIRecombination(T);
    }
    return TableRate_(s, s.tab_cold, GetTablePoint_(s, true, T, lnT),
                      kTabRecombCII_);
  }

  /// H2 formation on grains in cm^3 s^-1, including Z_d
  KOKKOS_INLINE_FUNCTION static Real KgrH2_(const NCRSettings& s, const Real T,
                                            const Real lnT) {
    const Real kgr = (s.interp_mode == kInterpExact)
                         ? rates::H2FormationGrainHM79(T)
                         : TableRate_(s, s.tab_cold,
                                      GetTablePoint_(s, true, T, lnT),
                                      kTabKgrH2_);
    return kgr * s.zd;
  }

  //-------------------------------------------------------------------------------------
  // Helpers
  /// The cell's radiation fields from the radiation functor's bands,
  /// Tigress's SetRadiationAndCosmicRayVariables without its local shielding
  KOKKOS_INLINE_FUNCTION static RadFields MakeRadFields_(
      const RegisterArray<Real, n_freq>& rad) {
    const Real chi_lw = rad[iph_LW];
    const Real chi_pe = rad[irad_GPE];
    RadFields fields;
    fields.chi_fuv = (chi_pe * kURadPEIsrf_ + chi_lw * kURadLWIsrf_) /
                     (kURadPEIsrf_ + kURadLWIsrf_);
    fields.chi_ci = rad[iph_C];
    fields.chi_co = chi_lw;
    fields.xi_cr = rad[irad_CR];
    fields.xi_diss_h2 = rad[iph_H2] * kXiDissH2Isrf_;
    return fields;
  }

  /// x_H2 as the rates see it, in [1e-20, 0.5]
  KOKKOS_INLINE_FUNCTION static Real FloorH2_(const Real xH2) {
    return Kokkos::fmin(Kokkos::fmax(xH2, kTiny_), 0.5);
  }

  /// x_H+ as the rates see it, in [1e-20, 1]
  KOKKOS_INLINE_FUNCTION static Real FloorHPlus_(const Real xH_plus) {
    return Kokkos::fmin(Kokkos::fmax(xH_plus, kTiny_), 1.0);
  }

  /// ln of the grain charging parameter psi = 1.7 chi_fuv sqrt(T) / n_e of
  /// Weingartner & Draine (2001), guarded against n_e = 0 and chi_fuv = 0
  KOKKOS_INLINE_FUNCTION static Real GrainChargingLnPsi_(const Real chi_fuv,
                                                         const Real T,
                                                         const Real ne) {
    return Kokkos::log(1.7 * chi_fuv * Kokkos::sqrt(T) / (ne + kTiny_) +
                       kTiny_);
  }

  /// Weight of the CIE side of the sigmoid between temp_hot0 and temp_hot1,
  /// Tigress's wgt2
  KOKKOS_INLINE_FUNCTION static Real HotWeight_(const NCRSettings& s,
                                                const Real T) {
    const Real T_mid = 0.5 * (s.temp_hot0 + s.temp_hot1);
    return 1.0 /
           (1.0 + Kokkos::exp(-10 * (T - T_mid) / (s.temp_hot1 - s.temp_hot0)));
  }

  //-------------------------------------------------------------------------------------
  // Chemistry rates
  /*!
   * \brief Grain-assisted recombination of H+, Weingartner & Draine (2001),
   * the alpha_gr of Tigress's HIIRates: blended out by the sigmoid between
   * temp_hot0 and temp_hot1 and zero above
   *
   * \param w_hot HotWeight_ at T, used if temp_hot0 < T < temp_hot1
   * \return Real The rate coefficient in cm^3 s^-1, including Z_d
   */
  KOKKOS_INLINE_FUNCTION static Real GrainRecombHII_(
      const NCRSettings& s, const RadFields& rad, const Real ne, const Real T,
      const Real lnT, const Real w_hot) {
    if (!(T < s.temp_hot1)) {
      return 0.0;
    }
    const Real lnPsi = GrainChargingLnPsi_(rad.chi_fuv, T, ne);
    Real alpha_gr = rates::GrainRecombHplus(lnT, lnPsi) * s.zd;
    if (T > s.temp_hot0) {
      alpha_gr *= 1.0 - w_hot;
    }
    return alpha_gr;
  }

  /*!
   * \brief H2 collisional dissociation by HI and H2, Glover & Mac Low
   * (2007), Tigress's GetCollDissRateH2. The critical density uses NCR's
   * 2*xH2/ncr_H2, which differs from GOW17's
   *
   * \details 10^(log10(a)*w) is computed as exp(ln(a)*w), algebraically
   * identical to Tigress
   *
   * \return Real The destruction rate per H2 in s^-1, zero at and below 700 K
   */
  KOKKOS_INLINE_FUNCTION static Real CollDissH2_(const Real nH, const Real T,
                                                 const Real lnT,
                                                 const Real xHI,
                                                 const Real xH2) {
    if (!(T > kTempCollDissH2_)) {
      return 0.0;
    }
    const Real logT4 = (lnT - kLn1e4_) * kLog10e_;
    const rates::H2CollDissCoeffs c = rates::H2CollDissGM07(T, lnT, logT4);
    const Real ncrinv =
        Kokkos::fmax(xHI / c.ncrHI + 2.0 * xH2 / c.ncrH2, kTiny_);
    const Real n2ncr = nH * ncrinv;
    const Real w_high = n2ncr / (1. + n2ncr);
    const Real w_low = 1. / (1. + n2ncr);
    const Real k_H2_HI = Kokkos::exp(c.ln_k9h * w_high + c.ln_k9l * w_low);
    const Real k_H2_H2 = Kokkos::exp(c.ln_k10h * w_high + c.ln_k10l * w_low);
    return k_H2_H2 * nH * xH2 + k_H2_HI * nH * xHI;
  }

  /*!
   * \brief Equilibrium C+ abundance, Tigress's GetxCII: CR, photo and
   * CR-induced photoionization against radiative, grain-assisted and
   * C+ + H2 recombination
   *
   * \return Real The C+ abundance per H
   */
  KOKKOS_INLINE_FUNCTION static Real GetxCII_(const NCRSettings& s,
                                              const RadFields& rad,
                                              const Real xe, const Real xH2,
                                              const Real nH, const Real T,
                                              const Real lnT) {
    // CR ionization
    const Real k_C_cr = 3.85 * rad.xi_cr;
    // photoionization, plus C-ionizing photons from CR-induced electronic
    // excitation of H2
    const Real k_C_photo = 3.5e-10 * rad.chi_ci + 520.0 * 2.0 * xH2 * rad.xi_cr;
    // radiative recombination
    const Real k_Cplus_e = RecombCII_(s, T, lnT);
    // grain assisted recombination
    const Real lnPsi = GrainChargingLnPsi_(rad.chi_fuv, T, nH * xe);
    const Real k_Cplus_gr = rates::GrainRecombCplus(lnT, lnPsi) * s.zd;
    // CH2+ formation channel, 3.3e-13 T^-1.3 exp(-23/T)
    const Real k_Cplus_H2 = 3.3e-13 * Kokkos::exp(-1.3 * lnT - 23. / T);
    const Real c = (k_C_cr + k_C_photo) / nH;
    const Real al = k_Cplus_e * xe + k_Cplus_gr + k_Cplus_H2 * xH2 + c;
    const Real ar = s.xC_tot * c;
    return ar / al;
  }

  /*!
   * \brief Electrons from molecular ions, Tigress's GetxeMol
   *
   * \return Real The electron abundance per H from molecular ions
   */
  KOKKOS_INLINE_FUNCTION static Real GetxeMol_(const NCRSettings& s,
                                               const RadFields& rad,
                                               const Real xe, const Real xH2,
                                               const Real nH, const Real T) {
    const Real k1620 = 1e-14 * s.zd;
    const Real k1622 = 1e-14 * s.zd;
    const Real k1621 = 1e-9;
    // 1e-7 (T/100)^-0.5
    const Real k1619 = 1.0e-7 / Kokkos::sqrt(T * 1e-2);
    const Real phi_s = (1.0 - xe / kxELMax_) * 0.67 / (1.0 + xe / 0.05);
    const Real xS = 5.3e-6 * s.zg;
    const Real A = k1619 * (1.0 + k1621 / k1622 * xS);
    const Real B = k1620 + k1621 * xS;
    return 2.0 * xH2 *
           (Kokkos::sqrt(B * B + 4.0 * A * rad.xi_cr * (1.0 + phi_s) / nH) -
            B) /
           (2.0 * k1619);
  }

  /*!
   * \brief The factor 1 / (1 + n_crit^2 / n_H^2) of the equilibrium CO
   * abundance, Tigress's CalculateCOAbundance, which depends only on the
   * cell's density and radiation
   *
   * \return Real The factor, in [0, 1]
   */
  KOKKOS_INLINE_FUNCTION static Real CONcritFactor_(const NCRSettings& s,
                                                    const RadFields& rad,
                                                    const Real nH) {
    const Real kcr16 = rad.xi_cr * 1.0e16;
    const Real term1 = Kokkos::fmax(4.0e3 * s.zd / (kcr16 * kcr16), 1.);
    const Real ncrit_exp = Kokkos::pow(rad.chi_co, ONE_3RD);
    // At xi_cr = 0 Tigress's form is inf * 0 = NaN. Use its limit there,
    // ncrit ~ kcr16^(1 - 2 chi_co^(1/3)), which is 0 or +inf
    const Real ncrit =
        (kcr16 > 0.0)
            ? Kokkos::pow(term1, ncrit_exp) *
                  (50. * kcr16 / Kokkos::pow(s.zd, 1.4))
            : Kokkos::pow(4.0e3 * s.zd, ncrit_exp) *
                  Kokkos::pow(0.0, 1.0 - 2.0 * ncrit_exp) * 50. /
                  Kokkos::pow(s.zd, 1.4);
    return 1.0 / (1.0 + (ncrit * ncrit) / (nH * nH));
  }

  /*!
   * \brief Equilibrium CO abundance, Tigress's CalculateCOAbundance
   *
   * \param co_ncrit_fac CONcritFactor_ of the cell
   * \return Real The CO abundance per H
   */
  KOKKOS_INLINE_FUNCTION static Real CalculateCOAbundance_(
      const NCRSettings& s, const Real co_ncrit_fac, const Real xH2,
      const Real xCII, const Real xOII) {
    return co_ncrit_fac * 2.0 * xH2 *
           Kokkos::fmin(s.xO_tot - xOII, s.xC_tot - xCII);
  }

  //-------------------------------------------------------------------------------------
  // Electrons and temperature
  /*!
   * \brief Bracketed, safeguarded root of an increasing residual g: the
   * Illinois variant of false position, which keeps the root bracketed
   *
   * \details If g(a) >= 0 the root is at or below a and a is returned. The
   * bracket is widened while g(b) < 0, by steps of -2 g(b), which suffices
   * when the slope of g is near 1, as for g(x_e) = x_e - F. Converges when an
   * iteration moves x by at most kXeRelTol_ * x, or the bracket is that
   * narrow, and returns the bracket midpoint after kXeMaxIter_ iterations.
   *
   * \param g The residual, increasing in x
   * \param a, b The bracket, a < b, normally with g(a) < 0 < g(b)
   * \param ga, gb g(a) and g(b), which the callers already have or need
   * \return Real The root
   */
  template <class Residual>
  KOKKOS_INLINE_FUNCTION static Real SolveBracketed_(const Residual& g, Real a,
                                                     Real b, Real ga, Real gb) {
    if (!(ga < 0.0)) {
      return a;
    }
    for (int n = 0; gb < 0.0 && n < kXeMaxWiden_; ++n) {
      const Real step = -2.0 * gb;
      a = b;
      ga = gb;
      b += step;
      gb = g(b);
    }
    if (!(gb > 0.0)) {
      return b;
    }

    // side of the bracket replaced last: -1 for a, +1 for b
    int side = 0;
    Real x_prev = b;
    for (int n = 0; n < kXeMaxIter_; ++n) {
      const Real x = (a * gb - b * ga) / (gb - ga);
      const Real gx = g(x);
      if (gx < 0.0) {
        a = x;
        ga = gx;
        // Illinois: halve the retained end when the same end moves twice
        if (side == -1) gb *= 0.5;
        side = -1;
      } else if (gx > 0.0) {
        b = x;
        gb = gx;
        if (side == 1) ga *= 0.5;
        side = 1;
      } else {
        return x;
      }
      if ((n > 0 && Kokkos::abs(x - x_prev) <= kXeRelTol_ * x) ||
          (b - a) <= kXeRelTol_ * x) {
        return x;
      }
      x_prev = x;
    }
    return 0.5 * (a + b);
  }

  /*!
   * \brief The electron abundance x_e = F(x_e, T), Tigress's
   * CalculateFreeElectron, solved jointly with the temperature T(x_e) =
   * E / C_v(x_e), or at a fixed temperature
   *
   * \details Below temp_hot0, F = x_H+ + x_C+,eq + x_O+ + x_e,mol (the cold
   * branch); at and above it, F = x_H+ + x_e,He(T) + Z_g x_e,metal(T) from
   * the CIE table (the hot branch). g(x_e) = x_e - F is increasing in x_e on
   * each branch, so each is solved by SolveBracketed_:
   *   - cold: bracket [x_H+ + x_O+, x_H+ + x_O+ + xC Z_g + x_e,mol(lo)], since
   *     x_C+,eq <= xC Z_g and x_e,mol only decreases as x_e increases and T
   *     decreases;
   *   - hot: bracket [x_H+, F(x_H+)].
   * The discontinuity at temp_hot0 is resolved deterministically: the cold
   * solution if its T < temp_hot0, otherwise the hot one. In the joint solve
   * T decreases as x_e increases, so if g_cold(hi) >= 0 the cold root is at
   * most hi and its T at least T(hi): the cold solve is skipped when T(hi)
   * is at or above temp_hot0, by kHotSkipMargin_ for the rounding of the
   * root, as its solution would be rejected. At a fixed T, T alone picks the
   * branch, as Tigress, and the hot branch is explicit.
   *
   * \param s The network settings
   * \param rad The radiation fields
   * \param nH The number density of H nuclei in cm^-3
   * \param xH2, xH_plus The floored abundances per H
   * \param E_ergs Joint solve only: the internal energy per H in erg
   * \param gamma Joint solve only: the adiabatic index
   * \param is_T_fixed Whether T is fixed at T_fixed rather than from E_ergs
   * \param T_fixed Fixed-temperature solve only: the temperature in K
   * \return ElectronSolution x_e, T and ln(T), and the cold branch's x_C+ at
   * them if it last evaluated the residual at x_e
   */
  KOKKOS_FUNCTION static ElectronSolution SolveElectrons_(
      const NCRSettings& s, const RadFields& rad, const Real nH,
      const Real xH2, const Real xH_plus, const Real E_ergs, const Real gamma,
      const bool is_T_fixed, const Real T_fixed) {
    // The temperature at x_e, before the rate floor
    auto temperature_raw = [&](const Real xe) {
      return is_T_fixed ? T_fixed
                        : E_ergs / Thermo::CvCold(xH2, xHe, xe, gamma);
    };
    // The temperature at x_e the rates see
    auto temperature = [&](const Real xe) {
      return Kokkos::fmax(temperature_raw(xe), kTempFloorRates_);
    };
    // O+ is tied to H+ by charge exchange
    const Real xO_plus = xH_plus * s.xO_tot;

    ElectronSolution sol;
    const bool try_cold =
        is_T_fixed ? (temperature(0.0) < s.temp_hot0) : true;
    if (try_cold) {
      // The last x_e the cold residual saw, never a solution until set, and
      // its ln(T) and x_C+, for the caller
      Real last_xe = -1.0;
      Real last_lnT = 0.0;
      Real last_xCII = 0.0;
      auto residual_cold = [&](const Real xe) {
        const Real T = temperature(xe);
        const Real lnT = Kokkos::log(T);
        const Real xCII = GetxCII_(s, rad, xe, xH2, nH, T, lnT);
        last_xe = xe;
        last_lnT = lnT;
        last_xCII = xCII;
        return xe - (xH_plus + xCII + xO_plus + GetxeMol_(s, rad, xe, xH2, nH, T));
      };
      const Real lo = xH_plus + xO_plus;
      const Real hi = lo + s.xC_tot +
                      GetxeMol_(s, rad, lo, xH2, nH, temperature(lo));
      const Real g_hi = residual_cold(hi);
      const bool is_cold_rejected =
          !is_T_fixed && !(g_hi < 0.0) &&
          temperature(hi) >= s.temp_hot0 * (1.0 + kHotSkipMargin_);
      if (!is_cold_rejected) {
        sol.xe = SolveBracketed_(residual_cold, lo, hi, residual_cold(lo), g_hi);
        sol.T_raw = temperature_raw(sol.xe);
        sol.T = Kokkos::fmax(sol.T_raw, kTempFloorRates_);
        if (is_T_fixed || sol.T < s.temp_hot0) {
          sol.has_xCII = (sol.xe == last_xe);
          sol.lnT = sol.has_xCII ? last_lnT : Kokkos::log(sol.T);
          sol.xCII = last_xCII;
          return sol;
        }
      }
    }

    // Hot branch, x_e of He and metals in collisional ionization equilibrium
    auto F_hot = [&](const Real xe) {
      const NCRCIETable::Point p = NCRCIETable::Locate(temperature(xe));
      return xH_plus + NCRCIETable::Interpolate(NCRCIETable::iXeHe, p) +
             s.zg * NCRCIETable::Interpolate(NCRCIETable::iXeMetal, p);
    };
    if (is_T_fixed) {
      sol.xe = F_hot(0.0);
    } else {
      auto residual_hot = [&](const Real xe) { return xe - F_hot(xe); };
      const Real hi = F_hot(xH_plus);
      sol.xe = SolveBracketed_(residual_hot, xH_plus, hi, xH_plus - hi,
                               residual_hot(hi));
    }
    sol.T_raw = temperature_raw(sol.xe);
    sol.T = Kokkos::fmax(sol.T_raw, kTempFloorRates_);
    sol.lnT = Kokkos::log(sol.T);
    sol.xCII = 0.0;
    sol.has_xCII = false;
    return sol;
  }

  /// The electron abundance at the fixed temperature T, for
  /// ElectronAbundance and PostSolve. xH2 and xH_plus are floored
  KOKKOS_INLINE_FUNCTION static Real ElectronsAtT_(const NCRSettings& s,
                                                   const RadFields& rad,
                                                   const Real nH,
                                                   const Real xH2,
                                                   const Real xH_plus,
                                                   const Real T) {
    return SolveElectrons_(s, rad, nH, xH2, xH_plus, 0.0, 0.0, true, T).xe;
  }

  //-------------------------------------------------------------------------------------
  // Cooling, in erg s^-1 H^-1
  /// Lyman alpha cooling, the Vrinceanu et al. (2014) refit of NCR, which
  /// differs from Thermo::CoolingLya's constants
  KOKKOS_INLINE_FUNCTION static Real CoolingLya_(const Real xe, const Real xHI,
                                                 const Real nH, const Real T,
                                                 const Real lnT) {
    const Real ne = xe * nH;
    const Real T4 = T * 1.0e-4;
    const Real lnT4 = lnT - kLn1e4_;
    // T4^0.14897 / (1 + (0.2 T4)^0.64897)
    const Real fac = 5.30856e-08 * Kokkos::exp(1.4897e-01 * lnT4) /
                     (1.0 + Kokkos::exp(0.64897 * (lnT4 - Thermo::ln5_)));
    const Real k01e = fac * Kokkos::exp(-11.84 / T4);
    const Real q01 = k01e * ne;
    const Real q10 = (Thermo::g0HI_ / Thermo::g1HI_) * fac * ne;
    return Thermo::Cooling2Level_(q01, q10, Thermo::A10HI_, Thermo::E10HI_,
                                  xHI);
  }

  /// HII cooling: free-free (Draine 2011 eq. 10.12) and case B recombination
  KOKKOS_INLINE_FUNCTION static Real CoolingHII_(const Real xe,
                                                 const Real xHII,
                                                 const Real nH, const Real T,
                                                 const Real lnT,
                                                 const Real alpha_B) {
    // Free-free, a = ln(3.9810717e-06 T)
    const Real a = lnT + kLn3_9810717em6_;
    const Real gff_T = 1.0 + 0.44 / (1.0 + 0.058 * a * a);
    const Real c_ff =
        1.422e-25 * gff_T * Kokkos::sqrt(T * 1e-4) * nH * xe * xHII;
    // Recombination
    const Real Err_B =
        (0.684 - 0.0208 * (lnT - kLn1e4_)) * units::Units::k_boltzmann_cgs * T;
    const Real c_rec = Err_B * alpha_B * nH * xe * xHII;
    return c_ff + c_rec;
  }

  /// Nebular metal line cooling, Tigress's CoolingNeb
  KOKKOS_INLINE_FUNCTION static Real CoolingNeb_(const Real xe,
                                                 const Real xHII,
                                                 const Real nH, const Real T,
                                                 const Real lnT,
                                                 const Real zg) {
    constexpr Real aNEB[7] = {-0.0050817, 0.00765822, 0.11832144, -0.50515842,
                              0.81569592, -0.58648172, 0.69170381};
    const Real lnT4 = lnT - kLn1e4_;
    const Real poly =
        (((((aNEB[0] * lnT4 + aNEB[1]) * lnT4 + aNEB[2]) * lnT4 + aNEB[3]) *
              lnT4 +
          aNEB[4]) *
             lnT4 +
         aNEB[5]) *
            lnT4 +
        aNEB[6];
    // pow(10, poly)
    const Real poly_fit = Kokkos::exp(Thermo::ln10_ * poly);
    // pow(ne/100, 0.38 - 0.12 lnT4)
    const Real f_red =
        1 / (1.0 + 0.12 * Kokkos::exp((0.38 - 0.12 * lnT4) *
                                      Kokkos::log(xe * nH * 1e-2)));
    return 3.677602203699553e-21 * zg * xHII * xe * nH / Kokkos::sqrt(T) *
           Kokkos::exp(-38585.52 / T) * poly_fit * f_red;
  }

  /*!
   * \brief Cooling by metals, nebular lines and recombination on PAHs,
   * Tigress's CoolingOther. C+, CO, CI, OI and O+ are in equilibrium
   */
  KOKKOS_INLINE_FUNCTION Real CoolingOther_(const ElectronSolution& es,
                                            const Real xHI, const Real xH2,
                                            const Real xHII) const {
    const Real xe = es.xe;
    const Real T = es.T;
    const Real lnT = es.lnT;
    const Real xOII = xHII * settings_.xO_tot;
    const Real xCII = es.has_xCII
                          ? es.xCII
                          : GetxCII_(settings_, rad_, xe, xH2, n_H, T, lnT);
    const Real xCO =
        CalculateCOAbundance_(settings_, co_ncrit_fac_, xH2, xCII, xOII);
    const Real xOI = Kokkos::fmax(settings_.xO_tot - xOII - xCO, kTiny_);
    const Real xCI = Kokkos::fmax(settings_.xC_tot - xCII - xCO, kTiny_);
    const Real nHI = xHI * n_H;
    const Real nH2 = xH2 * n_H;
    const Real ne = xe * n_H;

    // De-excitation rates, from one table point in a table mode
    ColliderRates k_CII;
    ThreeLevelRates k_OI, k_CI;
    if (settings_.interp_mode == kInterpExact) {
      const Real lnT2 = lnT - Thermo::ln100_;
      k_CII = Thermo::RatesCII(T, lnT2);
      k_OI = OIRates(lnT);
      k_CI = Thermo::RatesCI(T, lnT, lnT2);
    } else {
      const TablePoint p = GetTablePoint_(settings_, true, T, lnT);
      k_CII = ColdColliderRates_(settings_, p, kTabK10CII_);
      k_OI = ColdThreeLevelRates_(settings_, p, kTabOI_);
      k_CI = ColdThreeLevelRates_(settings_, p, kTabCI_);
    }

    // xCI >= 1e-20, so its cooling is always on, as in Tigress
    const Real c_CI = Thermo::CoolingThreeLevel_(k_CI, Thermo::CI_atom_, xCI,
                                                 nHI, nH2, ne, T);
    // CO rotational lines, with the constant velocity gradient NCR_dvdr and
    // NCR's maximum effective length
    const Real c_CO = (xCO != 0.0)
                          ? Thermo::CoolingCOLVG(xCO, nHI, nH2, ne, T,
                                                 xCO * n_H, settings_.dvdr,
                                                 kLeffCO_)
                          : 0.0;
    const Real c_CII = Thermo::CoolingCII(k_CII, xCII, nHI, nH2, ne, T);
    const Real c_OI = Thermo::CoolingThreeLevel_(k_OI, Thermo::OI_atom_, xOI,
                                                 nHI, nH2, ne, T);
    const Real c_neb = CoolingNeb_(xe, xHII, n_H, T, lnT, settings_.zg);
    const Real c_Rec = Thermo::CoolingRec(settings_.zd, T, ne, rad_.chi_fuv);

    return c_OI + c_neb + c_CI + c_CII + c_CO + c_Rec;
  }

  /*!
   * \brief The internal energy equation, Tigress's EvaluateThermal for the
   * default options, converted to code units
   *
   * \details Below temp_hot1: hydrogen, H2 and "other" cooling, and CR, PE
   * and H2 heating. At and above temp_hot0 the CIE He and metal cooling is
   * added, blended with the non-equilibrium "other" cooling and the heating
   * by the sigmoid up to temp_hot1, above which the heating is kHeatMin_.
   * Net cooling is switched off at or below the T/mu floor.
   *
   * \param es The electron solution: x_e, T before and after the rate floor,
   * ln(T) and maybe x_C+
   * \param alpha_B, k_coll_HI, kgr, xi_coll_H2 The rate coefficients
   * evaluate_function already has
   * \param w_hot HotWeight_ at T, used if temp_hot0 <= T < temp_hot1
   * \return Real dE/dt in code units
   */
  KOKKOS_INLINE_FUNCTION Real Edot_(const ElectronSolution& es, const Real xHI,
                                    const Real xH2, const Real xH_plus,
                                    const Real alpha_B, const Real k_coll_HI,
                                    const Real kgr, const Real xi_coll_H2,
                                    const Real w_hot) const {
    const Real xe = es.xe;
    const Real T = es.T;
    const Real lnT = es.lnT;
    const Real ne = xe * n_H;
    Real heat;
    Real cool_hyd = 0.0;
    Real cool_other = 0.0;

    // HI: Lyman alpha and collisional ionization. The latter always uses the
    // exact k_coll, as Tigress, which k_coll_HI is in exact mode
    if (T > kTempHI_) {
      const Real k_coll_exact = (settings_.interp_mode == kInterpExact)
                                    ? k_coll_HI
                                    : CollIonHI(T, lnT);
      cool_hyd += CoolingLya_(xe, xHI, n_H, T, lnT);
      cool_hyd +=
          kEvRatio_ * Thermo::CoolingHIion(xHI, xe, k_coll_exact * n_H);
    }
    // HII: free-free and recombination
    cool_hyd += CoolingHII_(xe, xH_plus, n_H, T, lnT, alpha_B);
    if (T < settings_.temp_hot1) {
      // H2 rovibrational lines and collisional dissociation, 4.48 eV per
      // dissociation, gated as Tigress. xH2 is floored, so Tigress's xH2 > 0
      // test is always true
      cool_hyd += Thermo::CoolingH2M21(xH2, xHI, n_H, T) +
                  4.48 * kEv_ * xH2 * xi_coll_H2;
      // Other cooling and heating. The H2 and CR heating are computed with
      // Thermo's eV_, corrected by kEvRatio_
      cool_other = CoolingOther_(es, xHI, xH2, xH_plus);
      const Real xi_diss = rad_.xi_diss_h2;
      const Real heat_eV =
          Thermo::HeatingH2gr(xHI, xH2, n_H, T, kgr * n_H, xi_diss) +
          Thermo::HeatingH2pump(xHI, xH2, n_H, T, xi_diss) +
          Thermo::HeatingH2diss(xi_diss, xH2) +
          Thermo::HeatingCr(xe, n_H, xHI, xH2, rad_.xi_cr);
      heat = Thermo::HeatingPE(rad_.chi_fuv, settings_.zd, T, ne) +
             kEvRatio_ * heat_eV;
    } else {
      heat = kHeatMin_;
    }

    // Blend of the non-equilibrium and CIE cooling
    Real cool = cool_hyd;
    if (T >= settings_.temp_hot0) {
      // log10(T) from the shared lnT
      const NCRCIETable::Point p = NCRCIETable::LocateLog10(lnT * kLog10e_);
      const Real cool_he_tbl =
          n_H * NCRCIETable::Interpolate(NCRCIETable::iLambdaHe, p);
      const Real cool_metal_tbl =
          n_H * (settings_.zg *
                 NCRCIETable::Interpolate(NCRCIETable::iLambdaMetal, p));
      if (T < settings_.temp_hot1) {
        const Real wgt1 = 1.0 - w_hot;
        cool += wgt1 * cool_other + cool_he_tbl + w_hot * cool_metal_tbl;
        heat *= wgt1;
      } else {
        cool += cool_he_tbl + cool_metal_tbl;
      }
    } else {
      cool += cool_other;
    }

    // T/mu floor: no net cooling at or below it
    const Real T_mu = es.T_raw * (1.0 + xHe + xe - xH2) / mu_H_;
    if (T_mu <= settings_.temp_mu_floor && heat < cool) {
      return 0.0;
    }

    // convert to code units
    return units_time_cgs * ((heat - cool) * n_H / units_energy_density_cgs);
  }
};  // class NCRNetwork
};  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_NCR_HPP_
