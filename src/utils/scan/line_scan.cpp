//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.cpp
//  \brief Implementation file for exclusive prefix sums of cell-centered fields
//  along cartisian directions.

#include "utils/scan/line_scan.hpp"

#include <algorithm>

namespace line_scan {
//----------------------------------------------------------------------------------------
// x1 half-lines: exclusive scan along i, across the threads of a team.
// One team per (m, n, k, j) line; the team's threads split the i range.

// void LineScanX1(int nmb, int nvar, int nvar_offset, const RegionIndcs& indcs,
//                 const DvceArray5D<Real>& q, const DvceArray6D<Real>& out) {
//   const int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
//   const int js = indcs.js, je = indcs.je;
//   const int ks = indcs.ks, ke = indcs.ke;

//   // 4D outer loop: teams over (m, n, k, j), scan over i inside
//   par_for_outer(
//       "line_scan_x1", DevExeSpace(), 0, 0, 0, (nmb - 1), 0, (nvar - 1), ks,
//       ke, js, je, KOKKOS_LAMBDA(TeamMember_t t, const int m, const int n,
//       const int k,
//                     const int j) {
//         const int nv = nvar_offset + n;
//         // MINUS_X1: exclusive prefix sum from the low face up to (not
//         // including) i
//         Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, is, ie + 1),
//                               [=](const int i, Real& update, const bool
//                               final) {
//                                 const Real x = q(m, n, k, j, i);
//                                 if (final) {
//                                   out(m, nv, MINUS_X1, k, j, i) = update;
//                                 }
//                                 update += x;
//                               });

//         // PLUS_X1: same scan with the index reversed, giving the suffix sum
//         to
//         // the high face
//         Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, nx1),
//                               [=](const int p, Real& update, const bool
//                               final) {
//                                 const int i = ie - p;
//                                 const Real x = q(m, n, k, j, i);
//                                 if (final) {
//                                   out(m, nv, PLUS_X1, k, j, i) = update;
//                                 }
//                                 update += x;
//                               });
//       });
// }

// //----------------------------------------------------------------------------------------
// // x2 half-lines: exclusive scan along j, serial within each thread.
// // One thread per (m, n, k, i) line; the thread walks its own line over j.

// void LineScanX2(int nmb, int nvar, int nvar_offset, const RegionIndcs& indcs,
//                 const DvceArray5D<Real>& q, const DvceArray6D<Real>& out) {
//   const int is = indcs.is, ie = indcs.ie;
//   const int js = indcs.js, je = indcs.je;
//   const int ks = indcs.ks, ke = indcs.ke;

//   // 4D par_for decomposes the flat index as (n,k,j,i) with i fastest; the
//   // ranges are chosen so those map to (m, n_var, k, i) and neighbouring
//   threads
//   // hold neighbouring i
//   par_for(
//       "line_scan_x2", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), ks, ke, is,
//       ie, KOKKOS_LAMBDA(const int m, const int n, const int k, const int i) {
//         const int nv = nvar_offset + n;
//         Real run = 0.0;
//         for (int j = js; j <= je; ++j) {
//           out(m, nv, MINUS_X2, k, j, i) = run;
//           run += q(m, n, k, j, i);
//         }
//         run = 0.0;
//         for (int j = je; j >= js; --j) {
//           out(m, nv, PLUS_X2, k, j, i) = run;
//           run += q(m, n, k, j, i);
//         }
//       });
// }

// //----------------------------------------------------------------------------------------
// // x3 half-lines: exclusive scan along k, serial within each thread.
// // One thread per (m, n, j, i) line; the thread walks its own line over k.

// void LineScanX3(int nmb, int nvar, int nvar_offset, const RegionIndcs& indcs,
//                 const DvceArray5D<Real>& q, const DvceArray6D<Real>& out) {
//   const int is = indcs.is, ie = indcs.ie;
//   const int js = indcs.js, je = indcs.je;
//   const int ks = indcs.ks, ke = indcs.ke;

//   // 4D par_for decomposes the flat index as (n,k,j,i) with i fastest; the
//   // ranges are chosen so those map to (m, n_var, j, i) and neighbouring
//   threads
//   // hold neighbouring i
//   par_for(
//       "line_scan_x3", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), js, je, is,
//       ie, KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
//         const int nv = nvar_offset + n;
//         Real run = 0.0;
//         for (int k = ks; k <= ke; ++k) {
//           out(m, nv, MINUS_X3, k, j, i) = run;
//           run += q(m, n, k, j, i);
//         }
//         run = 0.0;
//         for (int k = ke; k >= ks; --k) {
//           out(m, nv, PLUS_X3, k, j, i) = run;
//           run += q(m, n, k, j, i);
//         }
//       });
// }

LineScan::LineScan(MeshBlockPack* ppack, Direction direction, SumType sum_type)
    : pmy_pack(ppack), direction(direction), sum_type(sum_type) {
  auto& indcs = pmy_pack->pmesh->mb_indcs;

  // Compute the maximum number of meshblocks
  const int nmb =
      std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank);

  // Determine the total number of real cells and add 2 in the direction of
  // the scan to store the block wide sum and prefixes
  int ni = indcs.nx1;
  int nj = (indcs.nx2 > 1) ? indcs.nx2 : 1;
  int nk = (indcs.nx3 > 1) ? indcs.nx3 : 1;
  switch (direction) {
    case Direction::I:
      ni += 2;
      break;
    case Direction::J:
      nj += 2;
      break;
    case Direction::K:
      nk += 2;
      break;
  }

  // Allocate storage
  scan_data = DvceArray4D<Real>("scan_data", nmb, nk, nj, ni);
}
}  // namespace line_scan
