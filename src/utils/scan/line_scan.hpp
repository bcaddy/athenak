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

#include "athena.hpp"
#include "mesh/mesh.hpp"

namespace line_scan {
//----------------------------------------------------------------------------------------
/*!
 * \class LineScan
 * \brief Perform either a prefix or suffix sum along a given axis. All sums are
 * exclusive
 *
 */
class LineScan {
 public:
  // Direction along which the scan is performed.
  enum class Direction { I, J, K };
  const Direction direction;

  // Whether the sum is an exclusive prefix or suffix sum.
  enum class SumType { Prefix, Suffix };
  const SumType sum_type;

  explicit LineScan(MeshBlockPack* ppack, Direction direction,
                    SumType sum_type);
  ~LineScan() = default;

  MeshBlockPack* pmy_pack;

  // Number of cells in scan_data, including the 2 extra "ghost" in the scan
  // direction
  const int nmb, ni, nj, nk;

  // Real cell indices
  const int is, ie, js, je, ks, ke;

  // const so it can't be resized or reassigned, but its elements are writable.
  // Must be declared after nmb, ni, nj, nk since it is initialized from them
  const DvceArray4D<Real> scan_data;

  // Run the block local scan, primarily selects the proper function to run
  void BlockLocalScan();
  // The functions for running each scan in each direction
  // In place exclusive prefix sum along i of the real cells in each meshblock.
  // The block wide sum is stored in the upper ghost cell, ie+1
  void BlockLocalScan_I_Prefix() {
    // Local copies so the device lambda doesn't capture the host `this` pointer
    auto scan_data_ = scan_data;
    const int is_ = is, ni_ = ni;

    par_for_outer(
        "BlockLocalScan_I_Prefix", DevExeSpace(), 0, 0, 0, (nmb - 1), ks, ke, js, je,
        KOKKOS_LAMBDA(TeamMember_t t, const int m, const int k, const int j) {
          // Scan through the upper ghost cell, ni-1, so the exclusive prefix
          // stored there is the total sum along the line
          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(t, is_, ni_),
              [=](const int i, Real& update, const bool final) {
                // Read before writing since the scan is done in place
                const Real x = scan_data_(m, k, j, i);
                if (final) {
                  scan_data_(m, k, j, i) = update;
                }
                update += x;
              });
        });
  }
// void BlockLocalScan_J_Prefix();
// void BlockLocalScan_K_Prefix();
// void BlockLocalScan_I_Suffix();
// void BlockLocalScan_J_Suffix();
// void BlockLocalScan_K_Suffix();
};
}  // namespace line_scan
#endif  // UTILS_SCAN_LINE_SCAN_HPP_
