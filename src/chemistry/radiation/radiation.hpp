#ifndef CHEMISTRY_RADIATION_RADIATION_HPP_
#define CHEMISTRY_RADIATION_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief Interfaces to radiative transfer for the chemistry code

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry/network/chemistry_networks.hpp"
#include "chemistry/radiation/shielding.hpp"
#include "tasklist/task_list.hpp"
#include "units/units.hpp"
#include "utils/line_scan.hpp"
#include "utils/register_array.hpp"

namespace chemistry {
/*!
 * \class Radiation
 * \brief Computes the radiation field that the chemistry networks use
 */
class Radiation {
 public:
  Radiation(MeshBlockPack* ppack, ParameterInput* pin,
            const DvceArray5D<Real>& w0, const int species_start,
            const Real mu_H) {
    // Select the radiation integrator, name matches Athena++'s --chem_radiation
    const std::string integrator =
        pin->GetOrAddString("chemistry", "chem_radiation", "constant");

    // The number of radiation frequencies and the column density species of
    // the network in use
    int n_freq;
    ColumnDensitySpecies species;
    const std::string network = pin->GetString("chemistry", "network");
    if (network == "H2") {
      // H2 doesn't use radiation, leave ir unallocated
      return;
    } else if (network == "GOW17") {
      n_freq = GOW17Network::n_freq;
      species =
          GetColumnDensitySpecies<GOW17Network>(ppack, pin, species_start);
      zd_ = GOW17Network::GetSettings(pin, ppack).zd;
    } else {
      std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<chemistry>/network = " << network
                << " is not a valid option, choose from {H2, GOW17}"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // The incident radiation field, in units of the Draine 1978 field, and the
    // unshielded cosmic ray ionization rate in s^-1
    const Real G0 = pin->GetOrAddReal("chemistry", "radiation_G0", 1e-6);
    const Real cr_rate = pin->GetOrAddReal("chemistry", "radiation_CR", 2e-16);

    if (integrator == "constant") {
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
      // TODO: compute ir from the six-ray column densities
      // The incident field of each ray, G0 unless set for the face it enters
      // the domain through, as in Athena++
      G0_[BoundaryFace::inner_x1] =
          pin->GetOrAddReal("chemistry", "radiation_G0_inner_x1", G0);
      G0_[BoundaryFace::inner_x2] =
          pin->GetOrAddReal("chemistry", "radiation_G0_inner_x2", G0);
      G0_[BoundaryFace::inner_x3] =
          pin->GetOrAddReal("chemistry", "radiation_G0_inner_x3", G0);
      G0_[BoundaryFace::outer_x1] =
          pin->GetOrAddReal("chemistry", "radiation_G0_outer_x1", G0);
      G0_[BoundaryFace::outer_x2] =
          pin->GetOrAddReal("chemistry", "radiation_G0_outer_x2", G0);
      G0_[BoundaryFace::outer_x3] =
          pin->GetOrAddReal("chemistry", "radiation_G0_outer_x3", G0);
      cr_rate_ = cr_rate;
      is_cr_shielding_ =
          pin->GetOrAddBoolean("chemistry", "radiation_CR_shielding", false);
      col_unit_ = ppack->punit->density_cgs() * ppack->punit->length_cgs() /
                  (mu_H * ppack->punit->hydrogen_mass_cgs);
      is_ = ppack->pmesh->mb_indcs.is;
      js_ = ppack->pmesh->mb_indcs.js;
      ks_ = ppack->pmesh->mb_indcs.ks;
      scans = std::make_unique<SixRayScans>(ppack, pin, w0, species);
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
  /// The number of six-ray rays, one per face of the domain
  static constexpr int n_rays = 6;

  /// The network-specific inputs to ColumnDensityFunc
  struct ColumnDensitySpecies {
    Real xC;  ///< The total C abundance per H nucleon
    // The indices in w0 of the species the columns need, size_t like the
    // networks' species enums
    std::size_t iH2, iCO, iHCO_plus, iCHx, iC_plus;
  };

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
   * private type. The species are runtime values, from
   * GetColumnDensitySpecies, so that every network shares this functor.
   *
   * \tparam Dir The direction of the scan, which picks the cell width.
   */
  template <line_scan::Direction Dir>
  struct ColumnDensityFunc {
    DvceArray5D<Real> w0;          ///< The hydro or MHD primitives
    ColumnDensitySpecies species;  ///< The network's column density species

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
      const Real xH2 = w0(m, species.iH2, k, j, i);
      const Real xCO = w0(m, species.iCO, k, j, i);

      // Abundances only used to compute xCI
      const Real xHCO_plus = w0(m, species.iHCO_plus, k, j, i);
      const Real xCHx = w0(m, species.iCHx, k, j, i);
      const Real xC_plus = w0(m, species.iC_plus, k, j, i);

      // Neutral C from conservation, as GOW17's ghost species
      const Real xCI =
          Kokkos::fmax(species.xC - xHCO_plus - xCHx - xCO - xC_plus, 0.0);

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

  /// A LineScan of the column densities along one of the six rays
  template <line_scan::Direction Dir, line_scan::ScanKind Kind>
  using ColumnDensityScan =
      line_scan::LineScan<Dir, Kind, n_cols, ColumnDensityFunc<Dir>>;

  /*!
   * \brief The column density scans along all six rays, constructed and driven
   * in the same order on every rank as LineScan requires. Public since nvcc
   * rejects LineScan's kernels when a template argument is a private type.
   */
  struct SixRayScans {
    SixRayScans(MeshBlockPack* ppack, ParameterInput* pin,
                const DvceArray5D<Real>& w0,
                const ColumnDensitySpecies& species)
        : i_prefix(ppack, {w0, species}, pin, "chemistry"),
          i_suffix(ppack, {w0, species}, pin, "chemistry"),
          j_prefix(ppack, {w0, species}, pin, "chemistry"),
          j_suffix(ppack, {w0, species}, pin, "chemistry"),
          k_prefix(ppack, {w0, species}, pin, "chemistry"),
          k_suffix(ppack, {w0, species}, pin, "chemistry") {}

    ColumnDensityScan<line_scan::Direction::I, line_scan::ScanKind::Prefix>
        i_prefix;
    ColumnDensityScan<line_scan::Direction::I, line_scan::ScanKind::Suffix>
        i_suffix;
    ColumnDensityScan<line_scan::Direction::J, line_scan::ScanKind::Prefix>
        j_prefix;
    ColumnDensityScan<line_scan::Direction::J, line_scan::ScanKind::Suffix>
        j_suffix;
    ColumnDensityScan<line_scan::Direction::K, line_scan::ScanKind::Prefix>
        k_prefix;
    ColumnDensityScan<line_scan::Direction::K, line_scan::ScanKind::Suffix>
        k_suffix;
  };

  /*!
   * \brief The shielded radiation field the chemistry network sees in a cell,
   * from the six-ray column densities. Each ray's
   * incident field is attenuated by dust and, in the H2, CO and CI
   * photodissociation bands, by self-shielding
   * (ChemRadIntegrator::UpdateRadiation in chem_rad/integrators/six_ray.cpp),
   * then averaged over the six rays (ChemNetwork::InitializeNextStep in
   * chemistry/network/gow17.cpp).
   *
   * \details Self-contained, holding everything it reads by value or as a
   * View, so that it can be captured in any kernel. Either call it in the
   * chemistry update kernel, keeping the radiation field in registers, or in
   * a kernel of its own that writes the field out. Requires the six-ray scans
   * to be complete. A functor rather than a lambda since nvcc won't compile a
   * KOKKOS_LAMBDA returned from a function with a deduced return type. Public
   * since nvcc rejects kernels capturing a private type.
   *
   * \tparam Network_t The chemistry network, which sets the bands, and the
   * size of the field so it can be returned in registers.
   */
  template <typename Network_t>
  struct ShieldedRadiationFunc {
    /*!
     * \brief Take the column densities from rad's six-ray scans, the only
     * columns this reads, and the six-ray inputs rad read from <chemistry>
     *
     * \param rad The radiation, with <chemistry>/chem_radiation = six_ray so
     * that its scans exist
     */
    explicit ShieldedRadiationFunc(const Radiation& rad)
        : G0_(rad.G0_),
          cr_rate_(rad.cr_rate_),
          is_cr_shielding_(rad.is_cr_shielding_),
          zd_(rad.zd_),
          col_unit_(rad.col_unit_),
          is_(rad.is_),
          js_(rad.js_),
          ks_(rad.ks_) {
      columns_[BoundaryFace::inner_x1] = rad.scans->i_prefix.scan_data;
      columns_[BoundaryFace::outer_x1] = rad.scans->i_suffix.scan_data;
      columns_[BoundaryFace::inner_x2] = rad.scans->j_prefix.scan_data;
      columns_[BoundaryFace::outer_x2] = rad.scans->j_suffix.scan_data;
      columns_[BoundaryFace::inner_x3] = rad.scans->k_prefix.scan_data;
      columns_[BoundaryFace::outer_x3] = rad.scans->k_suffix.scan_data;
    }

    /*!
     * \param m The MeshBlock's index in the MeshBlockPack
     * \param k, j, i The cell's indices, including ghost cells
     * \return The radiation field in each band, averaged over the six rays
     */
    KOKKOS_INLINE_FUNCTION RegisterArray<Real, Network_t::n_freq> operator()(
        const int m, const int k, const int j, const int i) const {
      // Athena++'s constants
      const Real bH2 = 3.0e5;        // H2 velocity dispersion in cm/s
      const Real NH0_CR = 9.35e20;   // column above which CRs are attenuated
      const Real sigmaPE = 1.0e-21;  // Thermo::sigmaPE_, private there
      // A local copy, since nvcc can't read a static constexpr member by
      // reference in device code
      constexpr Kokkos::Array<Real, Network_t::n_ph> kph_avfac =
          Network_t::kph_avfac;

      // The sum over the rays of each band, zeroed explicitly rather than
      // relying on RegisterArray's value-initialization
      RegisterArray<Real, Network_t::n_freq> rad;
      for (int ifreq = 0; ifreq < Network_t::n_freq; ++ifreq) {
        rad[ifreq] = 0.0;
      }

      // The cell's indices in the columns, which index real cells from 0
      const int kc = k - ks_, jc = j - js_, ic = i - is_;

      for (int iang = 0; iang < n_rays; ++iang) {
        // Column densities in cm^-2
        const auto& col = columns_[iang];
        const Real NH = col(m, iNH, kc, jc, ic) * col_unit_;
        const Real NH2 = col(m, iNH2, kc, jc, ic) * col_unit_;
        const Real NCO = col(m, iNCO, kc, jc, ic) * col_unit_;
        const Real NC = col(m, iNC, kc, jc, ic) * col_unit_;
        const Real AV = NH * zd_ / 1.87e21;

        // H2, CO and CI self-shielding
        const Real fs_H2 = Shielding::fShield_H2(NH2, bH2);
        const Real fs_CO = Shielding::fShield_CO_V09(NCO, NH2);
        const Real fs_C = Shielding::fShield_C(NC, NH2);

        // Add this ray's field to the sum over the rays. Photo-reactions,
        // attenuated by dust and, in the H2, CO and CI bands, self-shielding
        for (int ifreq = 0; ifreq < Network_t::n_ph; ++ifreq) {
          Real ray = G0_[iang] * Kokkos::exp(-kph_avfac[ifreq] * AV);
          if (ifreq == Network_t::iph_H2) ray *= fs_H2;
          if (ifreq == Network_t::iph_CO) ray *= fs_CO;
          if (ifreq == Network_t::iph_C) ray *= fs_C;
          rad[ifreq] += ray;
        }
        // Photoelectric heating
        rad[Network_t::irad_GPE] +=
            G0_[iang] * Kokkos::exp(-NH * sigmaPE * zd_);
        // Cosmic rays. Tests NH <= NH0_CR as Athena++ does, so a NaN NH gives
        // a NaN rate
        if (is_cr_shielding_ && !(NH <= NH0_CR)) {
          rad[Network_t::irad_CR] += cr_rate_ * (NH0_CR / NH);
        } else {
          rad[Network_t::irad_CR] += cr_rate_;
        }
      }

      // Average over the rays
      for (int ifreq = 0; ifreq < Network_t::n_freq; ++ifreq) {
        rad[ifreq] /= n_rays;
      }
      return rad;
    }

   private:
    /// The column densities of each ray, the scan_data of its scan in
    /// SixRayScans, indexed by the BoundaryFace the ray enters the domain
    /// through. Private so that only the six-ray scans are read
    Kokkos::Array<DvceArray5D<Real>, n_rays> columns_;
    /// The incident radiation field of each ray, indexed as columns_, in units
    /// of the Draine 1978 field
    RegisterArray<Real, n_rays> G0_;
    Real cr_rate_;          ///< The unshielded cosmic ray ionization rate, s^-1
    bool is_cr_shielding_;  ///< Whether to attenuate the cosmic ray rate
    Real zd_;               ///< The dust metallicity relative to solar
    /// Converts the code unit columns to cm^-2,
    /// density_cgs * length_cgs / (mu_H * hydrogen_mass_cgs)
    Real col_unit_;
    /// The first real cell, since the columns index real cells from 0
    int is_, js_, ks_;
  };

  /// The six-ray column density scans, only allocated for six_ray
  std::unique_ptr<SixRayScans> scans;

  /*!
   * \brief Task that computes the column densities along all six rays into
   * the scans' scan_data, which is ready to read once this returns. Only
   * registered when the scans exist, i.e. <chemistry>/chem_radiation = six_ray.
   */
  TaskStatus ComputeColumnDensities(Driver* pdrive, int stage) {
    line_scan::MultiScanDriver(scans->i_prefix, scans->i_suffix,
                               scans->j_prefix, scans->j_suffix,
                               scans->k_prefix, scans->k_suffix);
    return TaskStatus::complete;
  }

 private:
  // The six-ray inputs, read in the constructor for ShieldedRadiationFunc. See
  // its members for their meanings
  RegisterArray<Real, n_rays> G0_;
  Real cr_rate_;
  bool is_cr_shielding_;
  Real zd_;
  Real col_unit_;
  int is_, js_, ks_;

  /*!
   * \brief Get Network_t's inputs to ColumnDensityFunc
   *
   * \tparam Network_t The chemistry network in use
   * \param ppack The MeshBlockPack the chemistry is on
   * \param pin The ParameterInput, from which the network's settings are read
   * \param species_start The index in w0 of the first chemistry scalar
   * \return ColumnDensitySpecies The network's column density species
   */
  template <typename Network_t>
  static ColumnDensitySpecies GetColumnDensitySpecies(MeshBlockPack* ppack,
                                                      ParameterInput* pin,
                                                      const int species_start) {
    return {Network_t::GetSettings(pin, ppack).xC,
            species_start + Network_t::IH2,
            species_start + Network_t::ICO,
            species_start + Network_t::IHCO_plus,
            species_start + Network_t::ICHx,
            species_start + Network_t::IC_plus};
  }
};

}  // namespace chemistry
#endif  // CHEMISTRY_RADIATION_RADIATION_HPP_
