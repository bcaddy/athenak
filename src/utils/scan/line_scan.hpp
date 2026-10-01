//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.hpp
//  \brief Header file for exclusive prefix sums of cell-centered fields along
//  cartisian directions.

#ifndef UTILS_SCAN_LINE_SCAN_HPP_
#define UTILS_SCAN_LINE_SCAN_HPP_

#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"

namespace line_scan {
// Direction along which the scan is performed. Outside of LineScan so it can be
// named without the class template arguments
enum class Direction { I, J, K };

// Whether the sum is an exclusive prefix or suffix sum. Outside of LineScan so
// it can be named without the class template arguments
enum class ScanKind { Prefix, Suffix };

//----------------------------------------------------------------------------------------
/*!
 * \class LineScan
 * \brief Perform either a prefix or suffix sum along a given axis. All sums are
 * exclusive
 *
 * \tparam ValueFunc Type of a device callable that returns the value to scan at
 * a cell. A KOKKOS_LAMBDA must have the signature
 * \code
 * KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) -> Real
 * \endcode
 * where m is the meshblock and k, j, i are MeshBlock (mb_indcs) indices, which
 * include ghost cells.
 */
template <typename ValueFunc>
class LineScan {
 public:
  const Direction direction;
  const ScanKind scan_kind;

  /*!
   * \param value_func The callable that computes the value to scan.
   */
  LineScan(MeshBlockPack* ppack, Direction direction, ScanKind scan_kind,
           const ValueFunc& value_func)
      : pmy_pack(ppack),
        direction(direction),
        scan_kind(scan_kind),
        value_func(value_func),
        // Total number of real cells plus 2 in the direction of the scan to
        // store the block wide sum and prefixes
        nmb(std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank)),
        ni(ppack->pmesh->mb_indcs.nx1 + ((direction == Direction::I) ? 2 : 0)),
        nj(ppack->pmesh->mb_indcs.nx2 + ((direction == Direction::J) ? 2 : 0)),
        nk(ppack->pmesh->mb_indcs.nx3 + ((direction == Direction::K) ? 2 : 0)),
        // Real cells start at 1 in the scan direction (index 0 and n-1 hold the
        // block wide sum and prefixes) and at 0 otherwise
        is((direction == Direction::I) ? 1 : 0),
        ie(is + ppack->pmesh->mb_indcs.nx1 - 1),
        js((direction == Direction::J) ? 1 : 0),
        je(js + ppack->pmesh->mb_indcs.nx2 - 1),
        ks((direction == Direction::K) ? 1 : 0),
        ke(ks + ppack->pmesh->mb_indcs.nx3 - 1),
        // Allocate storage
        scan_data("scan_data", nmb, nk, nj, ni) {}
  ~LineScan() = default;

  MeshBlockPack* pmy_pack;

  // The callable that computes the value to scan at each cell
  const ValueFunc value_func;

  // Number of cells in scan_data, including the 2 extra "ghost" in the scan
  // direction
  const int nmb, ni, nj, nk;

  // Real cell indices
  const int is, ie, js, je, ks, ke;

  // const so it can't be resized or reassigned, but its elements are writable.
  // Must be declared after nmb, ni, nj, nk since it is initialized from them
  const DvceArray4D<Real> scan_data;

  /*!
   * \brief Run the block local scan, selecting the function for the direction
   * and scan kind.
   */
  void BlockLocalScan() {
    // Based on direction and ScanKind call the proper function
    constexpr auto key = [](Direction c, ScanKind s) -> int {
      return (static_cast<int>(c) << 8) | static_cast<int>(s);
    };
    switch (key(direction, scan_kind)) {
      case key(Direction::I, ScanKind::Prefix):
        BlockLocalScan_I_Prefix();
        break;
      // case key(Direction::J, ScanKind::Prefix):
      //   BlockLocalScan_J_Prefix();
      //   break;
      // case key(Direction::K, ScanKind::Prefix):
      //   BlockLocalScan_K_Prefix();
      //   break;
      case key(Direction::I, ScanKind::Suffix):
        BlockLocalScan_I_Suffix();
        break;
        // case key(Direction::J, ScanKind::Suffix):
        //   BlockLocalScan_J_Suffix();
        //   break;
        // case key(Direction::K, ScanKind::Suffix):
        //   BlockLocalScan_K_Suffix();
        //   break;
    }
  }

  // ===== The functions for running each scan in each direction =====
  /*!
   * \brief Exclusive prefix sum along i of value_func in the real cells of each
   * meshblock, stored in scan_data. The block wide sum is stored in the upper
   * ghost cell, ie+1.
   */
  void BlockLocalScan_I_Prefix() {
    // Local copies so the device lambda doesn't capture the host `this` pointer
    auto scan_data_ = scan_data;
    auto value_func_ = value_func;
    const int is_ = is, ni_ = ni;

    // The offsets from scan_data indices to MeshBlock indices, which include
    // ghost cells
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const int ioff = indcs.is - is, joff = indcs.js - js, koff = indcs.ks - ks;

    par_for_outer(
        "BlockLocalScan_I_Prefix", DevExeSpace(), 0, 0, 0, (nmb - 1), ks, ke,
        js, je,
        KOKKOS_LAMBDA(TeamMember_t t, const int m, const int k, const int j) {
          // Scan through the upper ghost cell, ni-1, so the exclusive prefix
          // stored there is the total sum along the line
          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(t, is_, ni_),
              [=](const int i, Real& update, const bool final) {
                const Real x = value_func_(m, k + koff, j + joff, i + ioff);
                if (final) {
                  scan_data_(m, k, j, i) = update;
                }
                update += x;
              });
        });
  }
  // void BlockLocalScan_J_Prefix();
  // void BlockLocalScan_K_Prefix();

  /*!
   * \brief Exclusive suffix sum along i of value_func in the real cells of each
   * meshblock, stored in scan_data. The block wide sum is stored in the lower
   * ghost cell, is-1.
   */
  void BlockLocalScan_I_Suffix() {
    // Local copies so the device lambda doesn't capture the host `this` pointer
    auto scan_data_ = scan_data;
    auto value_func_ = value_func;
    const int ie_ = ie;

    // The offsets from scan_data indices to MeshBlock indices, which include
    // ghost cells
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const int ioff = indcs.is - is, joff = indcs.js - js, koff = indcs.ks - ks;

    par_for_outer(
        "BlockLocalScan_I_Suffix", DevExeSpace(), 0, 0, 0, (nmb - 1), ks, ke,
        js, je,
        KOKKOS_LAMBDA(TeamMember_t t, const int m, const int k, const int j) {
          // Scan with the index reversed, from ie down through the lower ghost
          // cell, 0, so the exclusive suffix stored there is the total sum
          // along the line
          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(t, 0, ie_ + 1),
              [=](const int p, Real& update, const bool final) {
                const int i = ie_ - p;
                const Real x = value_func_(m, k + koff, j + joff, i + ioff);
                if (final) {
                  scan_data_(m, k, j, i) = update;
                }
                update += x;
              });
        });
  }
  // void BlockLocalScan_J_Suffix();
  // void BlockLocalScan_K_Suffix();
};
}  // namespace line_scan
#endif  // UTILS_SCAN_LINE_SCAN_HPP_
