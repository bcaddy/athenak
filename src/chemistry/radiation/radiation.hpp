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

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry/network/chemistry_networks.hpp"
#include "utils/line_scan.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
/*!
 * \class Radiation
 * \brief Computes the radiation field that the chemistry networks use
 */
class Radiation {
 public:
  Radiation(MeshBlockPack* ppack, ParameterInput* pin) {
    // Select the radiation integrator, name matches Athena++'s --chem_radiation
    const std::string integrator =
        pin->GetOrAddString("chemistry", "chem_radiation", "constant");

    // The number of radiation frequencies of the network in use
    int n_freq;
    const std::string network = pin->GetString("chemistry", "network");
    if (network == "H2") {
      // H2 doesn't use radiation, leave ir unallocated
      return;
    } else if (network == "GOW17") {
      n_freq = GOW17Network::n_freq;
    } else {
      std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<chemistry>/network = " << network
                << " is not a valid option, choose from {H2, GOW17}"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if (integrator == "constant") {
      // Get parameters
      const Real G0 = pin->GetOrAddReal("chemistry", "radiation_G0", 1e-6);
      const Real cr_rate =
          pin->GetOrAddReal("chemistry", "radiation_CR", 2e-16);

      // Assign values in host array
      HostArray1D<Real> ir_host("chemistry_ir_host", n_freq);
      for (size_t i = 0; i < ir_host.size(); i++) {
        ir_host(i) = G0;
      }
      ir_host(ir_host.size() - 1) = cr_rate;

      // Copy to device
      Kokkos::realloc(ir, ir_host.size());
      Kokkos::deep_copy(ir, ir_host);
    } else if (integrator == "six_ray") {
      // TODO: six-ray radiation integrator using LineScan
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
   * private type. The species indices are runtime values so that every network
   * shares this functor, e.g. iH2 = species_start + Network_t::IH2.
   *
   * \tparam Dir The direction of the scan, which picks the cell width.
   */
  template <line_scan::Direction Dir>
  struct ColumnDensityFunc {
    DvceArray5D<Real> w0;  ///< The hydro or MHD primitives
    Real xC;               ///< The total C abundance per H nucleon
    // The indices in w0 of the species the columns need
    int iH2, iCO, iHCO_plus, iCHx, iC_plus;

    KOKKOS_INLINE_FUNCTION RegisterArray<Real, n_cols> operator()(
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
      const Real xH2 = w0(m, iH2, k, j, i);
      const Real xCO = w0(m, iCO, k, j, i);

      // Abundances only used to compute xCI
      const Real xHCO_plus = w0(m, iHCO_plus, k, j, i);
      const Real xCHx = w0(m, iCHx, k, j, i);
      const Real xC_plus = w0(m, iC_plus, k, j, i);

      // Neutral C from conservation, as GOW17's ghost species
      const Real xCI = Kokkos::fmax(xC - xHCO_plus - xCHx - xCO - xC_plus, 0.0);

      // Column density of H nuclei through the cell. The species columns are
      // this times their abundance
      const Real NH = w0(m, IDN, k, j, i) * dx;

      // Return the column densities
      RegisterArray<Real, n_cols> cols;
      cols[iNH] = NH;
      cols[iNH2] = NH * xH2;
      cols[iNCO] = NH * xCO;
      cols[iNC] = NH * xCI;
      return cols;
    }
  };

 private:
};

}  // namespace chemistry
#endif  // CHEMISTRY_RADIATION_RADIATION_HPP_
