#ifndef CHEMISTRY_NETWORK_GOW17_HPP_
#define CHEMISTRY_NETWORK_GOW17_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17.hpp
//  \brief The implementation for the struct for the GOW17 chemistry network

#include <limits>

#include "athena.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
struct GOW17Settings {
  /// If we're using an isothermal equation of state
  bool isothermal;
  /// Dust metallicity
  Real zd;
  /// He abundance per H
  Real xHe;
  /// C abundance at Z=1
  Real xC;
  /// O abundance at Z=1
  Real xO;
  /// Si abundance at Z=1
  Real xSi;
  /// Minimum temperature for reaction rates, also applied to energy equation
  Real temperature_min_rates;
  /// Temperature above which heating is turned off
  Real temperature_max_heating;
  /// Temperature below which cooling is turned off
  Real temperature_min_cooling;
  /// Cooling for neutral medium is capped at this temperature
  Real temperature_max_cooling_nm;
  /// maximum effective length for CO cooling in cm
  Real Leff_CO_max;
  /// Whether or not to use H2 rovibrational cooling
  bool H2_rovib_cooling;
};

/*!
 * \brief The class for the GOW17 network.
 *
 * \details All chemistry networks are required to have a handful of specific
 * things so that the ODE solvers have a common interface to work with. They
 * need:
 *   - A `neqs` variable that specifies the number of equations. That should be
 *     the number of species plus 1 for the internal energy .
 *   - `y` and `f` RegisterArray variables to hold the current state and the
 *     result of evaluating the equations respectively.
 *   - An `evaluate_function` method that computes `f` from `y`
 *
 * General notes:
 *   - enums indicting positive ions (like H+) use either "p" or "_plus" instead
 * of "+" since
 *     "+" isn't a legal character in that case
 */
class GOW17Network {
 public:
  KOKKOS_FUNCTION GOW17Network(GOW17Settings const settings, Real const density,
                               Real const density_cgs, Real const mu_H,
                               Real const gamma, Real const hydrogen_mass_cgs,
                               Real const units_time_cgs,
                               Real const units_energy_density_cgs)
      : n_H(density * density_cgs / (mu_H * hydrogen_mass_cgs)),
        gamma(gamma),
        units_time_cgs(units_time_cgs),
        units_energy_density_cgs(units_energy_density_cgs),
        isothermal(settings.isothermal),
        zd(settings.zd),
        xHe(settings.xHe),
        xC(settings.xC),
        xO(settings.xO),
        xSi(settings.xSi),
        temperature_min_rates(settings.temperature_min_rates),
        temperature_max_heating(settings.temperature_max_heating),
        temperature_min_cooling(settings.temperature_min_cooling),
        temperature_max_cooling_nm(settings.temperature_max_cooling_nm),
        Leff_CO_max(settings.Leff_CO_max),
        H2_rovib_cooling(settings.H2_rovib_cooling) {}

  // ----- Number of equations -----
  static constexpr int neqs = 13;
  static constexpr int n_ghosts = 6;

  /// If the network is using an isothermal equation of state
  const bool isothermal;

  // ----- Arrays to store ODE state -----
  RegisterArray<Real, neqs + n_ghosts> y;  // The current state
  RegisterArray<Real, neqs + n_ghosts> f;  // The results of evaluating the ODEs

  // ----- Species indices within the ODE system ------
  enum : size_t {
    IHE_plus,   // He+
    IOHx,       // OHx
    ICHx,       // CHx
    ICO,        // CO
    IC_plus,    // C+
    IHCO_plus,  // HCO+
    IH2,        // H2
    IH_plus,    // H+
    IH3_plus,   // H3+
    IH2_plus,   // H2+
    IO_plus,    // O+
    ISi_plus,   // Si+
    IIE,        // internal energy, must be last
    ISi_g,      // Si, ghost species
    IC_g,       // C, ghost species
    IO_g,       // O, ghost species
    IHe_g,      // He, ghost species
    Ie_g,       // e, ghost species
    IH_g,       // H, ghost species
  };

  // ----- Names, used for output, must be the same order as the enum -----
  static constexpr std::array<std::string_view, neqs - 1> species_names = {
      "He+", "OHx", "CHx", "CO",  "C+", "HCO+",
      "H2",  "H+",  "H3+", "H2+", "O+", "Si+"};

  // below are ghost species. The abundances of ghost species are recalculated
  // everytime by other species.
  static constexpr std::array<std::string_view, 6> ghost_species_names = {
      "Si_g", "C_g", "O_g", "He_g", "e_g", "H_g"};

  // ----- cell values -----
  Real const n_H;  // The number density of hydrogen
  Real const gamma;

  // ----- unit conversion factors -----
  Real const units_time_cgs;
  Real const units_energy_density_cgs;

  // ----- Metallicity factors -----
  const Real zd;   /// Dust metallicity
  const Real xHe;  /// He abundance per H
  const Real xC;   /// C abundance at Z=1
  const Real xO;   /// O abundance at Z=1
  const Real xSi;  /// Si abundance at Z=1

  // ----- Temperature Variables -----
  /// Minimum temperature for reaction rates, also applied to energy equation
  const Real temperature_min_rates;
  /// Temperature above which heating is turned off
  const Real temperature_max_heating;
  /// Temperature below which cooling is turned off
  const Real temperature_min_cooling;
  /// Cooling for neutral medium is capped at this temperature
  const Real temperature_max_cooling_nm;

  // ----- Reaction rate constants -----
  static constexpr Real k_gr = 3.0e-17;
  // xi_cr is the primary cosmic-ray ionization rate per H
  static constexpr Real xi_cr = 2.0e-16;
  static constexpr Real k_cr = 3.0 * xi_cr;

  // ----- Chemical Settings -----
  /// maximum effective length for CO cooling in cm
  const Real Leff_CO_max;
  /// Whether or not to use H2 rovibrational cooling
  const bool H2_rovib_cooling;

  // ----- Member Functions -----
  /*!
   * \brief Get the settings for the GOW17 network from the input file
   *
   * \param pin The ParameterInput object
   * \return GOW17Settings The settings for the GOW17 network
   */
  static GOW17Settings GetSettings(ParameterInput* pin) {
    // Get the parameters from input file
    GOW17Settings output;

    // Dust metallicity
    output.zd = pin->GetOrAddReal("chemistry", "GOW17_Z_d", 1.0);
    // Gas metallicity
    Real zg = pin->GetOrAddReal("chemistry", "GOW17_Z_g", 1.0);
    // He abundance per H
    output.xHe = pin->GetOrAddReal("chemistry", "GOW17_xHe", 0.1);
    // C abundance at Z=1
    output.xC = zg * pin->GetOrAddReal("chemistry", "GOW17_xC", 1.6e-4);
    // O abundance at Z=1
    output.xO = zg * pin->GetOrAddReal("chemistry", "GOW17_xO", 3.2e-4);
    // Si abundance at Z=1
    output.xSi = zg * pin->GetOrAddReal("chemistry", "GOW17_xSi", 1.7e-6);
    // Isothermal EOS?
    output.isothermal =
        pin->GetOrAddBoolean("chemistry", "GOW17_isothermal", false);

    Real inf = std::numeric_limits<Real>::infinity();
    output.temperature_min_rates =
        pin->GetOrAddReal("chemistry", "GO17_temperature_min_rates", inf);
    output.temperature_max_heating =
        pin->GetOrAddReal("chemistry", "GOW17_temperature_max_heating", inf);
    output.temperature_min_cooling =
        pin->GetOrAddReal("chemistry", "GOW17_temperature_min_cooling", 1.);
    output.temperature_max_cooling_nm = pin->GetOrAddReal(
        "chemistry", "GOW17_temperature_max_cooling_nm", 1.0e9);

    // Chemical Settings
    output.Leff_CO_max =
        pin->GetOrAddReal("chemistry", "GOW17_Leff_CO_max", 3.0e20);
    output.H2_rovib_cooling =
        pin->GetOrAddBoolean("chemistry", "GOW17_H2_rovib_cooling", true);

    return output;
  }

  /*!
   * \brief Compute the temperature in the cell
   *
   * \return Real The temperature in the cell
   */
  KOKKOS_FUNCTION
  Real Temperature() {
    // energy per hydrogen atom
    const Real E_ergs = y(IIE) * units_energy_density_cgs / n_H;

    // Temperature
    Real T = E_ergs / Thermo::CvCold(y[IH2], xHe, y[Ie_g], gamma);

    // apply temperature floor, incase of very small or negative energy
    if (T < temperature_min_rates) {
      T = temperature_min_rates;
    }

    return T;
  }

  /*!
   * \brief Compute the cooling term from the temperature
   *
   * \param T The temperature in the cell
   * \return Real The cooling term, i.e. how much the energy decreases. Note
   * that this is a positive value so the energy update should look like `E =
   * HeatingTerm() - CoolingTerm();`
   */
  KOKKOS_FUNCTION
  Real CoolingTerm(Real T) {
    // Check that the temperature is below the maximum allowed for neutral
    // medium. If above then set it to T_max_NM
    T = (T > temperature_max_cooling_nm) ? temperature_max_cooling_nm : T;

    // cut-off cooling at low temperature
    if (T < temperature_min_cooling) {
      return 0;
    }

    // C+ fine structure line
    Real cooling = Thermo::CoolingCII(y[IC_plus], n_H * y[IH_g], n_H * y[IH2],
                                      n_H * y[Ie_g], T);
    // CI fine structure line
    cooling += Thermo::CoolingCI(y[IC_g], n_H * y[IH_g], n_H * y[IH2],
                                 n_H * y[Ie_g], T);
    // OI fine structure line
    cooling += Thermo::CoolingOI(y[IO_g], n_H * y[IH_g], n_H * y[IH2],
                                 n_H * y[Ie_g], T);
    // cooling of hot gas: radiative cooling, free-free.
    cooling += Thermo::CoolingLya(y[IH_g], n_H * y[Ie_g], T);
    //  CO rotational lines
    //  Calculate effective CO column density
    const Real vth = Kokkos::sqrt(2. * units::Units::k_boltzmann_cgs * T /
                                  units::Units::CO_mass_cgs);
    const Real nCO = n_H * y[ICO];
    const Real grad_small_ = vth / Leff_CO_max;
    const Real gradeff = Kokkos::fmax(gradv_, grad_small_);
    const Real NCOeff = nCO / gradeff;
    cooling += Thermo::CoolingCOR(y[ICO], n_H * y[IH_g], n_H * y[IH2],
                                  n_H * y[Ie_g], T, NCOeff);
    // H2 vibration and rotation lines
    if (H2_rovib_cooling) {
      cooling +=
          Thermo::CoolingH2(y[IH2], n_H * y[IH_g], n_H * y[IH2], n_H * y[IHe_g],
                            n_H * y[IH_plus], n_H * y[Ie_g], T);
    }
    // dust thermo emission. Disabled because our simulation does not go to high
    // enough density (>~ 10^5 cm-3) for dust cooling to matter.
    // cooling += 0.;  // Thermo::CoolingDustTd(zd,  n_H, T, 10.);

    // recombination of e on PAHs
    cooling += Thermo::CoolingRec(zd, T, n_H * y[Ie_g], rad_[irad_GPE]);
    // collisional dissociation of H2
    cooling += Thermo::CoolingH2diss(y[IH_g], y[IH2], k2body_[i2body_H2_H],
                                     k2body_[i2body_H2_H2]);
    // collisional ionization of HI
    cooling += Thermo::CoolingHIion(y[IH_g], y[Ie_g], k2body_[i2body_H_e]);

    return cooling;
  }

  /*!
   * \brief Compute the heating term from the temperature.
   *
   * \param T The temperature in the cell
   * \return Real The heating term, i.e. how much the energy increases. The
   * energy update should look like `E = HeatingTerm() - CoolingTerm();`
   */
  KOKKOS_FUNCTION
  Real HeatingTerm(Real const& T) {
    if (T > temperature_max_heating) {
      return 0.0;
    }

    // Cosmic ray heating
    Real heating =
        Thermo::HeatingCr(y[Ie_g], n_H, y[IH_g], y[IH2], rad_[irad_CR]);

    // photo electric effect on dust
    heating += Thermo::HeatingPE(rad_[irad_GPE], zd, T, n_H * y[Ie_g]);

    // H2 formation on dust grains
    const Real k_xH2_photo = kph_[iph_H2];
    heating +=
        Thermo::HeatingH2gr(y[IH_g], y[IH2], n_H, T, kgr_[igr_H], k_xH2_photo);

    // H2 UV pumping
    heating += Thermo::HeatingH2pump(y[IH_g], y[IH2], n_H, T, k_xH2_photo);

    // H2 Photodissociation
    heating += Thermo::HeatingH2diss(k_xH2_photo, y[IH2]);

    return heating;
  }

  /*!
   * \brief Evaluate the internal energy equation
   *
   * \return Real The result of evaluating the internal energy equation
   */
  KOKKOS_FUNCTION
  Real Edot() {
    if (isothermal) {
      return 0.0;
    }

    const Real T = Temperature();

    static constexpr Real T_floor = 1.0;  // temperature floor for cooling
    if (T < T_floor) {
      return 0;
    } else {
      const Real dEdt = HeatingTerm(T) - CoolingTerm(T);
      // convert to code units
      return units_time_cgs * (dEdt * n_H / units_energy_density_cgs);
    }
  }

  /*!
   * \brief Compute the creation and destruction rates. These are used like
   * `f(i) = rate.creation(i) - y(i) * rate.destruction(i);`
   *
   * \return CDRates_t A struct containing the creation and destruction rate
   * arrays.
   */
  KOKKOS_FUNCTION
  CDRates_t<neqs - 1> CDRates() {
    CDRates_t<neqs - 1> rates;

    // energy per hydrogen atom
    const Real E_ergs = y(IIE) * units_energy_density_cgs / n_H;

    // Verify abundances are positive, finite, and not NaN valued
    for (size_t i = 0; i < y.size; i++) {
      // Verify positivity
      y[i] = Kokkos::max(y[i], 0.0);

      // Check if finite or NaN valued and set to 0 if that's the case
      y[i] = (Kokkos::isinf(y[i]) or Kokkos::isnan(y[i])) ? 0 : y[i];
    }

    UpdateRates_();

    // // cosmic ray reactions
    // for (int i = 0; i < n_cr_; i++) {
    //   rate = kcr_[i] * yprev[incr_[i]];
    //   ydotg[incr_[i]] -= rate;
    //   ydotg[outcr_[i]] += rate;
    // }

    // // 2body reactions
    // for (int i = 0; i < n_2body_; i++) {
    //   rate = k2body_[i] * yprev[in2body1_[i]] * yprev[in2body2_[i]];
    //   if (yprev[in2body1_[i]] < 0 && yprev[in2body2_[i]] < 0) {
    //     rate *= -1.;
    //   }
    //   ydotg[in2body1_[i]] -= rate;
    //   ydotg[in2body2_[i]] -= rate;
    //   ydotg[out2body1_[i]] += rate;
    //   ydotg[out2body2_[i]] += rate;
    // }

    // // photo reactions
    // for (int i = 0; i < n_ph_; i++) {
    //   rate = kph_[i] * yprev[inph_[i]];
    //   ydotg[inph_[i]] -= rate;
    //   ydotg[outph1_[i]] += rate;
    // }

    // // grain assisted reactions
    // for (int i = 0; i < n_gr_; i++) {
    //   rate = kgr_[i] * yprev[ingr_[i]];
    //   ydotg[ingr_[i]] -= rate;
    //   ydotg[outgr_[i]] += rate;
    // }

    // // set ydot to return
    // for (int i = 0; i < NSPECIES; i++) {
    //   // return in code units
    //   ydot[i] = ydotg[i] * pmy_mb_->pmy_mesh->punit->code_time_cgs;
    // }

    // Verify abundances are positive, finite, and not NaN valued
    for (size_t i = 0; i < f.size; i++) {
      // Verify positivity
      f[i] = Kokkos::max(f[i], 0.0);

      // Check if finite or NaN valued and set to 0 if that's the case
      f[i] = (Kokkos::isinf(f[i]) or Kokkos::isnan(f[i])) ? 0 : f[i];
    }

    // convert to code units
    for (size_t i = 0; i < neqs - 1; i++) {
      rates.creation(i) *= units_time_cgs;
      rates.destruction(i) *= units_time_cgs;
    }

    return rates;
  }

  /*!
   * \brief Setup the network for the next iteration of the ODE solver
   *
   */
  KOKKOS_FUNCTION
  void SetupNextStep() {
    // Set the ghost species
    ComputeGhostSpecies_();
  }

  /*!
   * \brief Computes `f` using the values in `y`
   */
  KOKKOS_FUNCTION
  void evaluate_function() {
    // ----- Setup for the next step -----
    SetupNextStep();

    // Set negative values to zero
    for (int i = 0; i < y.size; i++) {
      if (y(i) < 0) {
        y(i) = 0;
      }
    }

    // ----- Internal energy equation -----
    f(IIE) = Edot();

    // ----- Creation & Destruction Rates -----
    const auto rates = CDRates();

    // Compute the changes
    for (size_t i = 0; i < neqs - 1; i++) {
      f(i) = rates.creation(i) - y(i) * rates.destruction(i);
    }
  }

 private:
  // ----- Photo Reactions -----
  // Reaction rates in Drain 1978 field units.
  // Reactions are, in order:
  // (0) h nu + *C -> C+ + *e
  // (1) h nu + CH -> *C + *H
  // (2) h nu + CO -> *C + *O            --self-shielding and shielding by H2
  // (3) h nu + OH -> *O + *H
  // ----added in GO2012--------
  // (4) h nu + H2 -> *H + *H            --self- and dust shielding
  // ----Si, from UMIST12
  // (5) h nu + *Si -> Si+

  /// Reaction Rate enum
  enum : size_t { iph_C, iph_CHx, iph_CO, iph_OHx, iph_H2, iph_Si };

  // Constants
  static constexpr int n_ph_ = 6;
  static constexpr int n_freq_ = n_ph_ + 2;

  /// radiation field intensity
  RegisterArray<Real, n_freq_> rad_;
  /// enum for indexing into the rad_ array
  enum : size_t { irad_GPE = n_ph_, irad_CR };

  /// rates for photo-reactions in s^-1
  RegisterArray<Real, n_ph_> kph_;

  // ----- Grain Reactions -----
  // Grain assisted recombination of H, H2, C+ and H+
  // (0) *H + *H + gr -> H2 + gr
  // (1) H+ + *e + gr -> *H + gr
  // (2) C+ + *e + gr -> *C + gr
  // (3) He+ + *e + gr -> *He + gr
  // ------Si, from WD2001-----
  // (4) Si+ + *e + gr -> *Si + gr

  /// constants
  static constexpr int n_gr_ = 5;

  /// rates for grain assisted reactions in cm^3 s^-1 z_d^-1
  RegisterArray<Real, n_gr_> kgr_;
  /// enum for indexing into kgr_
  enum : size_t { igr_H, igr_Hp, igr_Cp, igr_Hep, igr_Sip };

  // ----- Chemical Network -----
  // clang-format off
  // 2 body reactions
  // NOTE: photons from recombination are ignored
  // Reactions are, in order.
  //  -- are equations of special rate treatment in Glover, Federrath+ 2010:
  // (0) H3+ + *C -> CH + H2         --Vissapragada2016 new rates
  // (1) H3+ + *O -> OH + H2
  // (2) H3+ + CO -> HCO+ + H2
  // (3) He+ + H2 -> H+ + *He + *H    --fit to Schauer1989
  // (4) He+ + CO -> C+ + *O + *He
  // (5) C+ + H2 -> CH + *H         -- schematic reaction for C+ + H2 -> CH2+
  // (6) C+ + OH -> HCO+             -- Schematic equation for C+ + OH -> CO+ + H.
  // Use rates in KIDA website.
  // (7) CH + *O -> CO + *H
  // (8) OH + *C -> CO + *H          --exp(0.108/T)
  // (9) He+ + *e -> *He             --(17) Case B
  // (10) H3+ + *e -> H2 + *H
  // (11) C+ + *e -> *C              -- Include RR and DR, Badnell2003, 2006.
  // (12) HCO+ + *e -> CO + *H
  // ----added in GO2012--------
  // (13) H2+ + H2 -> H3+ + *H       --(54) exp(-T/46600)
  // (14) H+ + *e -> *H              --(12) Case B
  // ---collisional dissociation, only important at high temperature T>1e3---
  // (15) H2 + *H -> 3 *H            --(9) Density dependent. See Glover+MacLow2007
  // (16) H2 + H2 -> H2 + 2 *H       --(10) Density dependent. See Glover+MacLow2007
  // (17) *H + *e -> H+ + 2 *e       --(11) Relates to Te
  // ----added for H3+ destruction in addition to (10)----
  // (18) H3+ + *e -> *3H            --(111)
  // ----added He+ destruction in addtion to (3), from UMIST12----
  // (19) He+ + H2 -> H2+ + *He
  // ----added CH reaction to match for abundances of CH---
  // (20) CH + *H -> H2 + *C
  // ----added to match the Meudon code ---
  // (21) OH + *O -> *O + *O + *H
  // ---branching of C+ + H2 ------
  // (22) C+ + H2 + *e -> *C + *H + *H
  // ---Si , rate from UMIST12---
  // (23) Si+ + *e -> *Si
  // --- H2O+ + e reaction ---
  // (24) H3+ + *O + *e -> H2 + *O + *H
  // --- OH destruction with He+
  // (25) He+ + OH -> O+ + *He + *H
  // --- H2+ charge exchange with H ---
  // (26) H2+ + *H -> H+ + H2
  //  --- O+ reactions ---
  // (27) H+ + *O -> O+ + *H -- exp(-232/T)
  // (28) O+ + *H -> H+ + *O
  // (29) O+ + H2 -> OH + *H     -- branching of H2O+
  // (30) O+ + H2 -> *O + *H + *H  -- branching of H2O+
  // clang-format on

  static constexpr int n_2body_ = 31;
  /// rates for 2 body reactions in s^-1 cm^3
  RegisterArray<Real, n_2body_> k2body_;
  /// enum for indexing into k2body_
  enum : size_t {
    i2body_H3p_C,       // index for H3+ + *C -> CH + H2 reaction
    i2body_H3p_O,       // index for H3+ + *O -> OH + H2 reaction
    i2body_H3p_CO,      // index for H3+ + CO -> HCO+ + H2 reaction
    i2body_Hep_H2,      // index for He+ + H2 -> H+ + *He + *H reaction
    i2body_Hep_CO,      // index for He+ + CO -> C+ + *O + *He reaction
    i2body_Cp_H2,       // index for C+ + H2 -> CH + *H reaction
    i2body_Cp_OH,       // index for C+ + OH -> HCO+ reaction
    i2body_CH_O,        // index for CH + *O -> CO + *H reaction
    i2body_OH_C,        // index for OH + *C -> CO + *H reaction
    i2body_Hep_e,       // index for He+ + *e -> *He reaction
    i2body_H3p_e,       // index for H3+ + *e -> H2 + *H reaction
    i2body_Cp_e,        // index for C+ + *e -> *C reaction
    i2body_HCOp_e,      // index for HCO+ + *e -> CO + *H reaction
    i2body_H2p_H2,      // index for H2+ + H2 -> H3+ + *H reaction
    i2body_Hp_e,        // index for H+ + *e -> *H reaction
    i2body_H2_H,        // index for H2 + *H -> 3 *H reaction
    i2body_H2_H2,       // index for H2 + H2 -> H2 + 2 *H reaction
    i2body_H_e,         // index for *H + *e -> H+ + 2 *e reaction
    i2body_H3p_e_3H,    // index for H3+ + *e -> *3H reaction
    i2body_Hep_H2_H2p,  // index for He+ + H2 -> H2+ + *He reaction
    i2body_CH_H,        // index for CH + *H -> H2 + *C reaction
    i2body_OH_O,        // index for OH + *O -> *O + *O + *H reaction
    i2body_Cp_H2_e,     // index for C+ + H2 + *e -> *C + *H + *H reaction
    i2body_Sip_e,       // index for Si+ + *e -> *Si reaction
    i2body_H3p_O_H2,    // index for H3+ + *O + *e -> H2 + *O + *H reaction
    i2body_Hep_OH,      // index for He+ + OH -> O+ + *He + *H reaction
    i2body_H2p_H,       // index for H2+ + *H -> H+ + H2 reaction
    i2body_Hp_O,        // index for H+ + *O -> O+ + *H reaction
    i2body_Op_H,        // index for O+ + *H -> H+ + *O reaction
    i2body_Op_H2_OH,    // index for O+ + H2 -> OH + *H reaction
    i2body_Op_H2        // index for O+ + H2 -> *O + *H + *H reaction
  };

  // parameters related to CO cooling
  // these are needed for LVG approximation
  Real gradv_;  // absolute value of velocity gradient in cgs, >0

  KOKKOS_FUNCTION
  void ComputeGhostSpecies_() {
    // set the ghost species
    f[ISi_g] = xSi - y[ISi_plus];
    f[IC_g] = xC - y[IHCO_plus] - y[ICHx] - y[ICO] - y[IC_plus];
    f[IO_g] = xO - y[IHCO_plus] - y[IOHx] - y[ICO] - y[IO_plus];
    f[IHe_g] = xHe - y[IHE_plus];
    f[Ie_g] = y[IHE_plus] + y[IC_plus] + y[IHCO_plus] + y[IH3_plus] +
              y[IH2_plus] + y[IH_plus] + y[IO_plus] + y[ISi_plus];
    f[IH_g] = 1.0 - (y[IOHx] + y[ICHx] + y[IHCO_plus] + 3.0 * y[IH3_plus] +
                     2.0 * y[IH2_plus] + y[IH_plus] + 2.0 * y[IH2]);
  }
};  // class GOW17Network
};  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_GOW17_HPP_
