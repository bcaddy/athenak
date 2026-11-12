#ifndef CHEMISTRY_RADIATION_RADIATION_HPP_
#define CHEMISTRY_RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief Interfaces to radiative transfer for the chemistry code

#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry/network/chemistry_networks.hpp"
#include "utils/line_scan.hpp"

namespace chemistry {
/*!
 * \class Radiation
 * \brief The radiation field the chemistry network sees
 *
 * \tparam Network_t The chemistry network. Only GOW17 uses radiation; for the
 * other networks ir is left empty.
 */
template <typename Network_t>
class Radiation {
  /// Whether the network uses radiation
  static constexpr bool has_radiation = std::is_same_v<Network_t, GOW17Network>;

 public:
  Radiation(MeshBlockPack* ppack, ParameterInput* pin) {
    // Select the radiation integrator, name matches Athena++'s --chem_radiation
    const std::string integrator =
        pin->GetOrAddString("chemistry", "chem_radiation", "constant");

    if (integrator == "constant") {
      if constexpr (has_radiation) {
        // Get parameters
        const Real G0 = pin->GetOrAddReal("chemistry", "radiation_G0", 1e-6);
        const Real cr_rate =
            pin->GetOrAddReal("chemistry", "radiation_CR", 2e-16);

        // Assign values in host array
        HostArray1D<Real> ir_host("chemistry_ir_host", Network_t::n_freq);
        for (size_t i = 0; i < ir_host.size(); i++) {
          ir_host(i) = G0;
        }
        ir_host(ir_host.size() - 1) = cr_rate;

        // Copy to device
        Kokkos::realloc(ir, ir_host.size());
        Kokkos::deep_copy(ir, ir_host);
      }
    } else if (integrator == "six_ray") {
      if constexpr (has_radiation) {
        // TODO: six-ray radiation integrator using LineScan
      } else {
        std::cerr << "### FATAL ERROR in " << __FILE__ << " at line "
                  << __LINE__ << std::endl
                  << "<chemistry>/chem_radiation = six_ray requires a network "
                     "that uses radiation, i.e. <chemistry>/network = GOW17"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<chemistry>/chem_radiation = " << integrator
                << " is not a valid option, choose from {constant, six_ray}"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  ~Radiation() {}

  DvceArray1D<Real> ir;

  /// Index of each column density field in the six-ray LineScans
  enum : int { iNH, iNH2, iNCO, iNC, n_cols };

  /*!
   * \brief The value_func the six-ray LineScans sum: each cell's contribution
   * to the H nuclei, H2, CO, and neutral C column densities along Dir, in code
   * units (mass density times length). Multiply by
   * density_cgs * length_cgs / (mu_H * hydrogen_mass_cgs) for cm^-2. The
   * contributions are extensive, as LineScan requires with mesh refinement.
   *
   * \details A functor rather than a lambda since nvcc won't compile a
   * KOKKOS_LAMBDA returned from a function with a deduced return type. Public
   * since nvcc rejects LineScan's kernels when a template argument is a
   * private type.
   *
   * \tparam Dir The direction of the scan, which picks the cell width.
   */
  template <line_scan::Direction Dir>
  struct ColumnDensityFunc {
    DvceArray5D<Real> w0;  ///< The hydro or MHD primitives
    int species_start;     ///< The index in w0 of the first chemistry scalar
    Real xC;               ///< The total C abundance per H nucleon

    KOKKOS_INLINE_FUNCTION Kokkos::Array<Real, n_cols> operator()(
        const int m, const int k, const int j, const int i,
        const RegionSize& size) const {
      // Cell width along the scan
      Real dx;
      if constexpr (Dir == line_scan::Direction::I) {
        dx = size.dx1;
      } else if constexpr (Dir == line_scan::Direction::J) {
        dx = size.dx2;
      } else if constexpr (Dir == line_scan::Direction::K) {
        dx = size.dx3;
      }

      // Abundances
      const Real xH2 = w0(m, species_start + Network_t::IH2, k, j, i);
      const Real xCO = w0(m, species_start + Network_t::ICO, k, j, i);

      // Abundances only used to compute xCI
      const Real xHCO_plus =
          w0(m, species_start + Network_t::IHCO_plus, k, j, i);
      const Real xCHx = w0(m, species_start + Network_t::ICHx, k, j, i);
      const Real xC_plus = w0(m, species_start + Network_t::IC_plus, k, j, i);

      // Neutral C from conservation, as GOW17's ghost species
      const Real xCI = Kokkos::fmax(xC - xHCO_plus - xCHx - xCO - xC_plus, 0.0);

      // Column density of H nuclei through the cell. The species columns are
      // this times their abundance
      const Real NH = w0(m, IDN, k, j, i) * dx;

      // Return the column densities
      return {NH, NH * xH2, NH * xCO, NH * xCI};
    }
  };

 private:
};

/// The radiation for each chemistry network, one of which is in use
using RadiationVariant =
    std::variant<Radiation<H2Network>, Radiation<GOW17Network>>;

/*!
 * \brief Construct the radiation for the network in use
 *
 * \param ppack The MeshBlockPack the chemistry is on
 * \param pin The ParameterInput, which selects the network
 * \return RadiationVariant The radiation, for <chemistry>/network
 */
inline RadiationVariant MakeRadiation(MeshBlockPack* ppack,
                                      ParameterInput* pin) {
  const std::string network = pin->GetString("chemistry", "network");
  if (network == "H2") {
    return RadiationVariant(std::in_place_type<Radiation<H2Network>>, ppack,
                            pin);
  } else if (network == "GOW17") {
    return RadiationVariant(std::in_place_type<Radiation<GOW17Network>>, ppack,
                            pin);
  }
  std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "<chemistry>/network = " << network
            << " is not a valid option, choose from {H2, GOW17}" << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace chemistry
#endif  // CHEMISTRY_RADIATION_RADIATION_HPP_
