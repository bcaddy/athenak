#ifndef CHEMISTRY_NETWORK_GOW17_HPP_
#define CHEMISTRY_NETWORK_GOW17_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gow17.hpp
//  \brief The implementation for the struct for the GOW17 chemistry network

#include <Kokkos_NumericTraits.hpp>

#include "athena.hpp"
#include "chemistry/chemistry_utils.hpp"
#include "chemistry/thermo/thermo.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
struct GOW17Settings {
  /// If we're using an isothermal equation of state
  bool isothermal;
  /// The temperature to use for an isothermal EOS
  Real isothermal_temperature;
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
  /// Maximum temperature for reaction rates
  Real temperature_max_rates;
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
  // H2 formation rate on grains
  bool is_kgrH2_const;
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
                               DvceArray1D<Real> ir, Real const density_cgs,
                               Real const mu_H, Real const gamma,
                               Real const hydrogen_mass_cgs,
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
        temperature_max_rates(settings.temperature_max_rates),
        temperature_max_heating(settings.temperature_max_heating),
        temperature_min_cooling(settings.temperature_min_cooling),
        temperature_max_cooling_nm(settings.temperature_max_cooling_nm),
        Leff_CO_max(settings.Leff_CO_max),
        H2_rovib_cooling(settings.H2_rovib_cooling),
        isothermal_temperature_(settings.isothermal_temperature),
        rad_(ir),
        is_kgrH2_const(settings.is_kgrH2_const) {}

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
  /// Maximum temperature for reaction rates. Does not apply for collisional
  /// dissociation reactions and energy equation
  const Real temperature_max_rates;
  /// Temperature above which heating is turned off
  const Real temperature_max_heating;
  /// Temperature below which cooling is turned off
  const Real temperature_min_cooling;
  /// Cooling for neutral medium is capped at this temperature
  const Real temperature_max_cooling_nm;

  // ----- Chemical Settings -----
  /// maximum effective length for CO cooling in cm
  const Real Leff_CO_max;
  /// Whether or not to use H2 rovibrational cooling
  const bool H2_rovib_cooling;
  // H2 formation rate on grains
  const bool is_kgrH2_const;

  // ----- Member Functions -----
  /*!
   * \brief Get the settings for the GOW17 network from the input file
   *
   * \param pin The ParameterInput object
   * \return GOW17Settings The settings for the GOW17 network
   */
  static GOW17Settings GetSettings(ParameterInput* pin, MeshBlockPack* ppack) {
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
    if (output.isothermal) {
      const Real mu_iso = pin->GetReal("chemistry", "GOW17_mu_iso");
      const Real cs = pin->GetReal("hydro", "iso_sound_speed");
      const Real temperature_mu_cgs =
          ppack->punit->pressure_cgs() / ppack->punit->density_cgs() *
          units::Units::hydrogen_mass_cgs / units::Units::k_boltzmann_cgs;
      output.isothermal_temperature = cs * cs * mu_iso * temperature_mu_cgs;
    } else {
      output.isothermal_temperature =
          Kokkos::Experimental::signaling_NaN_v<Real>;
    }

    Real inf = Kokkos::Experimental::infinity_v<Real>;
    output.temperature_min_rates =
        pin->GetOrAddReal("chemistry", "GO17_temperature_min_rates", 1.0);
    output.temperature_max_rates =
        pin->GetOrAddReal("chemistry", "GO17_temperature_max_rates", inf);
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
    // H2 formation rate on grains
    output.is_kgrH2_const =
        pin->GetOrAddBoolean("chemistry", "is_kgrH2_const", false);

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
    cooling += Thermo::CoolingRec(zd, T, n_H * y[Ie_g], rad_(irad_GPE));
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
        Thermo::HeatingCr(y[Ie_g], n_H, y[IH_g], y[IH2], rad_(irad_CR));

    // photo electric effect on dust
    heating += Thermo::HeatingPE(rad_(irad_GPE), zd, T, n_H * y[Ie_g]);

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

    // Verify abundances are positive, finite, and not NaN valued
    for (size_t i = 0; i < y.size; i++) {
      // Verify positivity
      y[i] = Kokkos::fmax(y[i], 0.0);

      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(y[i]) || Kokkos::isnan(y[i])) {
        Kokkos::abort("Error: NaN or Inf value found in GOW17 `y` array\n");
      }
    }

    UpdateRates_();

// cosmic ray reactions
#pragma unroll
    for (int i = 0; i < n_cr_; i++) {
      const Real rate = kcr_[i] * y[incr_[i]];
      f[incr_[i]] -= rate;
      f[outcr_[i]] += rate;
    }

    // 2body reactions
    constexpr size_t in2body1_[n_2body_] = {
        IH3_plus, IH3_plus, IH3_plus, IHE_plus, IHE_plus, IC_plus,   IC_plus,
        ICHx,     IOHx,     IHE_plus, IH3_plus, IC_plus,  IHCO_plus, IH2_plus,
        IH_plus,  IH2,      IH2,      IH_g,     IH3_plus, IHE_plus,  ICHx,
        IOHx,     IC_plus,  ISi_plus, IH3_plus, IHE_plus, IH2_plus,  IH_plus,
        IO_plus,  IO_plus,  IO_plus};
    constexpr size_t in2body2_[n_2body_] = {
        IC_g, IO_g, ICO,  IH2,  ICO,  IH2,  IOHx, IO_g, IC_g, Ie_g, Ie_g,
        Ie_g, Ie_g, IH2,  Ie_g, IH_g, IH2,  Ie_g, Ie_g, IH2,  IH_g, IO_g,
        IH2,  Ie_g, IO_g, IOHx, IH_g, IO_g, IH_g, IH2,  IH2};
    // Note: output to ghost species doesn't matter. The abundances of ghost
    // species are updated using the other species at every timestep
    constexpr size_t out2body1_[n_2body_] = {
        ICHx, IOHx,    IHCO_plus, IH_plus,  IC_plus, ICHx,     IHCO_plus, ICO,
        ICO,  IHe_g,   IH2,       IC_g,     ICO,     IH3_plus, IH_g,      IH_g,
        IH2,  IH_plus, IH_g,      IH2_plus, IH2,     IO_g,     IC_g,      ISi_g,
        IH2,  IO_plus, IH_plus,   IO_plus,  IH_plus, IOHx,     IO_g};
    constexpr size_t out2body2_[n_2body_] = {
        IH2,  IH2,  IH2,  IHe_g, IO_g, IH_g, IH_g, IH_g, IH_g,  IH_g, IH_g,
        IH_g, IH_g, IH_g, IH_g,  IH_g, IH_g, Ie_g, IH_g, IHe_g, IC_g, IH_g,
        IH_g, IH_g, IO_g, IHe_g, IH2,  IH_g, IO_g, IH_g, IH_g};
    for (int i = 0; i < n_2body_; i++) {
      Real rate = k2body_[i] * y[in2body1_[i]] * y[in2body2_[i]];
      if (y[in2body1_[i]] < 0 && y[in2body2_[i]] < 0) {
        rate *= -1.;
      }
      f[in2body1_[i]] -= rate;
      f[in2body2_[i]] -= rate;
      f[out2body1_[i]] += rate;
      f[out2body2_[i]] += rate;
    }

    // photo reactions
    constexpr size_t inph_[n_ph_] = {IC_g, ICHx, ICO, IOHx, IH2, ISi_g};
    constexpr size_t outph1_[n_ph_] = {IC_plus, IC_g, IC_g,
                                       IO_g,    IH_g, ISi_plus};
    for (int i = 0; i < n_ph_; i++) {
      const Real rate = kph_[i] * y[inph_[i]];
      f[inph_[i]] -= rate;
      f[outph1_[i]] += rate;
    }

    // grain assisted reactions
    constexpr size_t ingr_[n_gr_] = {IH_g, IH_plus, IC_plus, IHE_plus,
                                     ISi_plus};
    constexpr size_t outgr_[n_gr_] = {IH2, IH_g, IC_g, IHe_g, ISi_g};
    for (int i = 0; i < n_gr_; i++) {
      const Real rate = kgr_[i] * y[ingr_[i]];
      f[ingr_[i]] -= rate;
      f[outgr_[i]] += rate;
    }

    // Verify abundances are positive, finite, and not NaN valued
    for (size_t i = 0; i < f.size; i++) {
      // Verify positivity
      f[i] = Kokkos::fmax(f[i], 0.0);

      // Check if inf or NaN valued and throw abort if that's the case
      if (Kokkos::isinf(f[i]) || Kokkos::isnan(f[i])) {
        Kokkos::abort("Error: NaN or Inf value found in GOW17 `f` array\n");
      }
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
  /// The temperature to use for an isothermal EOS
  const Real isothermal_temperature_;
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
  // RegisterArray<Real, n_freq_> rad_;
  DvceArray1D<Real> rad_;
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
  // cosmic ray chemistry network
  // (0) cr + H2 -> H2+ + *e
  // (1) cr + *He -> He+ + *e
  // (2) cr + *H  -> H+ + *e
  // -----added as Clark + Glover 2015----
  // (3) cr + *C -> C+ + *e     --including direct and cr induce photo reactions
  // (4) crphoto + CO -> *O + *C
  // (5) cr + CO -> HCO+ + e  --schematic for cr + CO -> CO+ + e
  // -----Si, CR induced photo ionization, experimenting----
  // (6) cr + Si -> Si+ + e, UMIST12
  static constexpr int n_cr_ = 7;
  /// rates for cosmic-ray reactions  in s^-1
  RegisterArray<Real, n_cr_> kcr_;
  /// reactant index
  const int incr_[n_cr_] = {IH2, IHe_g, IH_g, IC_g, ICO, ICO, ISi_g};
  /// product index
  const int outcr_[n_cr_] = {IH2_plus, IHE_plus,  IH_plus, IC_plus,
                             IO_g,     IHCO_plus, ISi_plus};

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
  // ----added He+ destruction in addition to (3), from UMIST12----
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

  //----------------------------------------------------------------------------------------
  /*!
   * \brief Update the rates for chemical reactions.
   */
  KOKKOS_FUNCTION
  void UpdateRates_() {
    // energy per hydrogen atom
    const Real E_ergs = y(IIE) * units_energy_density_cgs / n_H;

    Real T;
    // constant or evolve temperature
    if (isothermal) {
      // isohermal EOS
      T = isothermal_temperature_;
    } else {
      T = E_ergs / Thermo::CvCold(y[IH2], xHe, y[Ie_g], gamma);
    }

    // cap T above some minimum temperature
    Real T_collisional = T;
    if (T < temperature_min_rates) {
      T = temperature_min_rates;
      T_collisional = T;
    } else if (T > temperature_max_rates) {
      // do not put upper limit on the temperature for collisional ionization
      // and dissociation rates T_collisional
      T = temperature_max_rates;
    }

    const Real logT = Kokkos::log10(T);
    const Real logT4coll = Kokkos::log10(T_collisional / 1.0e4);
    const Real lnTecoll = Kokkos::log(T_collisional * 8.6173e-5);

    Real ncr, n2ncr;
    Real psi;        // H+ grain recombination parameter
    Real kcr_H_fac;  // ratio of total rate to primary rate
    Real psi_gr_fac_;
    const Real kida_fac = (0.62 + 45.41 / Kokkos::sqrt(T)) * n_H;
    Real t1_CHx, t2_CHx;

    // cosmic ray reactions
    /// coefficients of rates relative to H in s^-1
    constexpr Real kcr_base_[n_cr_] = {2.0, 1.1, 1.0, 520., 92., 6.52, 8400.};
#pragma unroll
    for (int i = 0; i < n_cr_; i++) {
      kcr_[i] = kcr_base_[i] * rad_[irad_CR];
    }
    // cosmic ray induced photo-reactions, proportional to x(H2)
    //  (0) cr + H2 -> H2+ + *e
    //  (1) cr + *He -> He+ + *e
    //  (2) cr + *H  -> H+ + *e
    //  (3) cr + *C -> C+ + *e     --including direct and cr induce photo
    //  reactions (4) crphoto + CO -> *O + *C (5) cr + CO -> HCO+ + e
    //  --schematic for cr + CO -> CO+ + e (6) cr + Si -> Si+ + e, UMIST12
    kcr_H_fac = 1.15 * 2 * y[IH2] + 1.5 * y[IH_g];
    kcr_[0] *= kcr_H_fac;
    kcr_[2] *= kcr_H_fac;
    kcr_[3] *= (2 * y[IH2] + 3.85 / kcr_base_[3]);
    kcr_[4] *= 2 * y[IH2];
    kcr_[6] *= 2 * y[IH2];
    // 2 body reactions
    constexpr Real k2Texp[n_2body_] = {
        0.0,  -0.190, 0.0,    0.0, 0.0, -1.3, 0.0, 0.0,   -0.339, -0.5, -0.52,
        0.0,  -0.64,  0.042,  0.0, 0.0, 0.0,  0.0, -0.52, 0.0,    0.26, 0.0,
        -1.3, -0.62,  -0.190, 0.0, 0.0, 0.0,  0.0, 0.0,   0.0};
    constexpr Real k2body_base[n_2body_] = {
        1.00,    1.99e-9,  1.7e-9,    1.26e-13, 1.6e-9,        3.3e-13 * 0.7,
        1.00,    7.0e-11,  7.95e-10,  1.0e-11,  4.54e-7,       1.00,
        1.06e-5, 1.76e-9,  2.753e-14, 1.00,     1.00,          1.00,
        8.46e-7, 7.20e-15, 2.81e-11,  3.5e-11,  3.3e-13 * 0.3, 1.46e-10,
        1.99e-9, 1.00,     6.4e-10,   1.00,     1.00,          1.6e-9,
        1.6e-9};
#pragma unroll
    for (int i = 0; i < n_2body_; i++) {
      k2body_[i] = k2body_base[i] * Kokkos::pow(T, k2Texp[i]) * n_H;
    }

    // Special treatment of rates for some equations
    // (0) H3+ + *C -> CH + H2         --Vissapragada2016 new rates
    // rates for H3+ + C forming CH+ and CH2+
    constexpr Real A_kCHx_ = 1.04e-9;
    constexpr Real n_kCHx_ = 2.31e-3;
    constexpr Real c_kCHx_[4] = {3.4e-8, 6.97e-9, 1.31e-7, 1.51e-4};
    constexpr Real Ti_kCHx_[4] = {7.62, 1.38, 2.66e1, 8.11e3};
    t1_CHx = A_kCHx_ * Kokkos::pow(300. / T, n_kCHx_);
    t2_CHx = c_kCHx_[0] * Kokkos::exp(-Ti_kCHx_[0] / T) +
             c_kCHx_[1] * Kokkos::exp(-Ti_kCHx_[1] / T) +
             c_kCHx_[2] * Kokkos::exp(-Ti_kCHx_[2] / T) +
             c_kCHx_[3] * Kokkos::exp(-Ti_kCHx_[3] / T);
    k2body_[0] *= t1_CHx + Kokkos::pow(T, -1.5) * t2_CHx;
    // (3) He+ + H2 -> H+ + *He + *H   --fit to Schauer1989
    k2body_[3] *= Kokkos::exp(-22.5 / T);
    // (5) C+ + H2 -> CH + *H         -- schematic reaction for C+ + H2 -> CH2+
    k2body_[5] *= Kokkos::exp(-23. / T);
    // ---branching of C+ + H2 ------
    // (22) C+ + H2 + *e -> *C + *H + *H
    k2body_[22] *= Kokkos::exp(-23. / T);
    // (6) C+ + OH -> HCO+         -- Schematic equation for C+ + OH -> CO+ + H.
    // Use rates in KIDA website.
    k2body_[6] = 9.15e-10 * kida_fac;
    // (8) OH + *C -> CO + *H          --exp(0.108/T)
    k2body_[8] *= Kokkos::exp(0.108 / T);
    // (9) He+ + *e -> *He             --(17) Case B
    k2body_[9] *= 11.19 + (-1.676 + (-0.2852 + 0.04433 * logT) * logT) * logT;
    // (11) C+ + *e -> *C              -- Include RR and DR, Badnell2003, 2006.
    k2body_[11] = CII_rec_rate_(T) * n_H;
    // (13) H2+ + H2 -> H3+ + *H       --(54) exp(-T/46600)
    k2body_[13] *= Kokkos::exp(-T / 46600.);
    // (14) H+ + *e -> *H              --(12) Case B
    k2body_[14] *= Kokkos::pow(315614.0 / T, 1.5) *
                   Kokkos::pow(1.0 + Kokkos::pow(115188.0 / T, 0.407), -2.242);
    //--- H2O+ + e branching--
    // (1) H3+ + *O -> OH + H2
    // (24) H3+ + *O + *e -> H2 + *O + *H
    Real h2oplus_ratio, fac_H2Oplus_H2, fac_H2Oplus_e;
    // small number
    constexpr Real small_real = 1e-50;
    if (y[Ie_g] < small_real) {
      h2oplus_ratio = 1.0e10;
    } else {
      h2oplus_ratio = 6e-10 * y[IH2] / (5.3e-6 / Kokkos::sqrt(T) * y[Ie_g]);
    }
    fac_H2Oplus_H2 = h2oplus_ratio / (h2oplus_ratio + 1.);
    fac_H2Oplus_e = 1. / (h2oplus_ratio + 1.);
    k2body_[1] *= fac_H2Oplus_H2;
    k2body_[24] *= fac_H2Oplus_e;
    // (25) He+ + OH -> *H + *He + *O(O+)
    k2body_[25] = 1.35e-9 * kida_fac;
    //  --- O+ reactions ---
    //  (27) H+ + *O -> O+ + *H -- exp(-227/T)
    //  (28) O+ + *H -> H+ + *O
    //  (29) O+ + H2 -> OH + *H     -- branching of H2O+
    //  (30) O+ + H2 -> *O + *H + *H  -- branching of H2O+ */
    k2body_[27] *=
        (1.1e-11 * Kokkos::pow(T, 0.517) + 4.0e-10 * Kokkos::pow(T, 6.69e-3)) *
        Kokkos::exp(-227. / T);
    k2body_[28] *=
        4.99e-11 * Kokkos::pow(T, 0.405) + 7.5e-10 * Kokkos::pow(T, -0.458);
    k2body_[29] *= fac_H2Oplus_H2;
    k2body_[30] *= fac_H2Oplus_e;

    // Collisional dissociation, k>~1.0e-30 at T>~5e2.
    Real k9l, k9h, k10l, k10h, ncrH, ncrH2, div_ncr;
    constexpr Real temp_coll = 7.0e2;
    if (T_collisional > temp_coll && n_H > small_real) {
      // (15) H2 + *H -> 3 *H
      // (16) H2 + H2 -> H2 + 2 *H
      // --(9) Density dependent. See Glover+MacLow2007
      k9l = 6.67e-12 * Kokkos::sqrt(T_collisional) *
            Kokkos::exp(-(1. + 63590. / T_collisional));
      k9h = 3.52e-9 * Kokkos::exp(-43900.0 / T_collisional);
      k10l = 5.996e-30 * Kokkos::pow(T_collisional, 4.1881) /
             Kokkos::pow((1.0 + 6.761e-6 * T_collisional), 5.6881) *
             Kokkos::exp(-54657.4 / T_collisional);
      k10h = 1.3e-9 * Kokkos::exp(-53300.0 / T_collisional);
      ncrH = Kokkos::pow(
          10, (3.0 - 0.416 * logT4coll - 0.327 * logT4coll * logT4coll));
      ncrH2 = Kokkos::pow(
          10, (4.845 - 1.3 * logT4coll + 1.62 * logT4coll * logT4coll));
      div_ncr = y[IH_g] / (ncrH + small_real) + y[IH2] / (ncrH2 + small_real);
      if (div_ncr < small_real) {
        ncr = 1. / small_real;
      } else {
        ncr = 1. / div_ncr;
      }
      n2ncr = n_H / ncr;
      k2body_[15] = Kokkos::pow(10, Kokkos::log10(k9h) * n2ncr / (1. + n2ncr) +
                                        Kokkos::log10(k9l) / (1. + n2ncr)) *
                    n_H;
      k2body_[16] = Kokkos::pow(10, Kokkos::log10(k10h) * n2ncr / (1. + n2ncr) +
                                        Kokkos::log10(k10l) / (1. + n2ncr)) *
                    n_H;
      // (17) *H + *e -> H+ + 2 *e       --(11) Relates to Te
      k2body_[17] *= Kokkos::exp(
          -3.271396786e1 +
          (1.35365560e1 +
           (-5.73932875 +
            (1.56315498 +
             (-2.877056e-1 +
              (3.48255977e-2 +
               (-2.63197617e-3 +
                (1.11954395e-4 + (-2.03914985e-6) * lnTecoll) * lnTecoll) *
                   lnTecoll) *
                  lnTecoll) *
                 lnTecoll) *
                lnTecoll) *
               lnTecoll) *
              lnTecoll);  // NOLINT
    } else {
      k2body_[15] = 0.;
      k2body_[16] = 0.;
      k2body_[17] = 0.;
    }

    // photo reactions
    constexpr Real kph_base_[n_ph_] = {3.5e-10, 9.1e-10, 2.4e-10,
                                       3.8e-10, 5.7e-11, 4.5e-9};
#pragma unroll
    for (int i = 0; i < n_ph_; i++) {
      kph_[i] = kph_base_[i] * rad_[i];
    }

    // Grain assisted recombination of H and H2
    //   (0) *H + *H + gr -> H2 + gr
    kgr_[0] = get_kgr_H2_(T) * n_H * zd;
    //   (1) H+ + *e + gr -> *H + gr
    //   (2) C+ + *e + gr -> *C + gr
    //   (3) He+ + *e + gr -> *He + gr
    //   (4) Si+ + *e + gr -> *Si + gr
    //   , rate dependent on e abundance.
    if (y[Ie_g] > small_real) {
      constexpr Real cHp_[7] = {12.25,    8.074e-6, 1.378,   5.087e2,
                                1.586e-2, 0.4723,   1.102e-5};
      constexpr Real cCp_[7] = {45.58,    6.089e-3, 1.128,   4.331e2,
                                4.845e-2, 0.8120,   1.333e-4};
      constexpr Real cHep_[7] = {5.572,    3.185e-7, 1.512,   5.115e3,
                                 3.903e-7, 0.4956,   5.494e-7};
      constexpr Real cSip_[7] = {2.166,    5.678e-8, 1.874,   4.375e4,
                                 1.635e-6, 0.8964,   7.538e-5};
      // set lower limit to radiation field in calculating kgr_ to avoid nan
      // values.
      Real GPE_limit = 1.0e-10;
      Real GPE0 = rad_[irad_GPE];
      if (GPE0 < GPE_limit) {
        GPE0 = GPE_limit;
      }
      psi_gr_fac_ = 1.7 * GPE0 * Kokkos::sqrt(T) / n_H;
      psi = psi_gr_fac_ / y[Ie_g];
      kgr_[1] =
          1.0e-14 * cHp_[0] /
          (1.0 +
           cHp_[1] * Kokkos::pow(psi, cHp_[2]) *
               (1.0 +
                cHp_[3] * Kokkos::pow(T, cHp_[4]) *
                    Kokkos::pow(psi, -cHp_[5] - cHp_[6] * Kokkos::log(T)))) *
          n_H * zd;
      kgr_[2] =
          1.0e-14 * cCp_[0] /
          (1.0 +
           cCp_[1] * Kokkos::pow(psi, cCp_[2]) *
               (1.0 +
                cCp_[3] * Kokkos::pow(T, cCp_[4]) *
                    Kokkos::pow(psi, -cCp_[5] - cCp_[6] * Kokkos::log(T)))) *
          n_H * zd;
      kgr_[3] =
          1.0e-14 * cHep_[0] /
          (1.0 +
           cHep_[1] * Kokkos::pow(psi, cHep_[2]) *
               (1.0 +
                cHep_[3] * Kokkos::pow(T, cHep_[4]) *
                    Kokkos::pow(psi, -cHep_[5] - cHep_[6] * Kokkos::log(T)))) *
          n_H * zd;
      kgr_[4] =
          1.0e-14 * cSip_[0] /
          (1.0 +
           cSip_[1] * Kokkos::pow(psi, cSip_[2]) *
               (1.0 +
                cSip_[3] * Kokkos::pow(T, cSip_[4]) *
                    Kokkos::pow(psi, -cSip_[5] - cSip_[6] * Kokkos::log(T)))) *
          n_H * zd;
    } else {
      for (int i = 1; i < 5; i++) {
        kgr_[i] = 0.;
      }
    }
  }

  //----------------------------------------------------------------------------------------
  /*!
   * \brief Calculate the rate for CII recombination
   *
   * \param T The temperature
   * \return Real The rate for CII recombination
   */
  KOKKOS_FUNCTION
  Real CII_rec_rate_(const Real T) {
    constexpr Real A = 2.995e-9;
    constexpr Real B = 0.7849;
    constexpr Real T0 = 6.670e-3;
    constexpr Real T1 = 1.943e6;
    constexpr Real C = 0.1597;
    constexpr Real T2 = 4.955e4;
    const Real BN = B + C * Kokkos::exp(-T2 / T);
    const Real term1 = Kokkos::sqrt(T / T0);
    const Real term2 = Kokkos::sqrt(T / T1);
    const Real alpha_rr = A / (term1 * Kokkos::pow(1.0 + term1, 1.0 - BN) *
                               Kokkos::pow(1.0 + term2, 1.0 + BN));
    const Real alpha_dr =
        Kokkos::pow(T, -3.0 / 2.0) * (6.346e-9 * Kokkos::exp(-1.217e1 / T) +
                                      9.793e-09 * Kokkos::exp(-7.38e1 / T) +
                                      1.634e-06 * Kokkos::exp(-1.523e+04 / T));
    return (alpha_rr + alpha_dr);
  }

  //----------------------------------------------------------------------------------------
  /*!
   * \brief H2 formation rate on dust grains. TIGRESS-NCR (Kim+23)
   * implementation.
   *
   * \param T The temperature
   * \return Real H2 formation rate on dust grains
   */
  KOKKOS_FUNCTION
  Real get_kgr_H2_(const Real T) {
    Real kgr;
    const Real kgr0 = 3.0e-17;
    if (is_kgrH2_const) {
      // Use temperature independent rate
      kgr = kgr0;
    } else {
      // Use temperature dependent rate from Hollenbach & McKee (1979)
      // Taking Tgr2 = 0 and renormalized to have kgr ~ kgr_H2 near 200>T>50
      const Real T2 = T * 1e-2;
      kgr = kgr0 * Kokkos::sqrt(T2) * 2.0 /
            (1 + 0.4 * Kokkos::sqrt(T2) + 0.2 * T2 + 0.08 * T2 * T2);
    }
    return kgr;
  }
};  // class GOW17Network
};  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_GOW17_HPP_
