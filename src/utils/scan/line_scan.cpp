//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.cpp
//  \brief Stage A/C of a physical-space line scan. NOT a complete feature on
//  its own.
//
//  !!! INCOMPLETE !!! These kernels only produce the WITHIN-MeshBlock
//  contribution. Every line is truncated at the MeshBlock boundary, so on any
//  mesh with more than one block along a scan axis the output is NOT the
//  physical-space half-line sum. The cross-block / cross-rank carry-in term is
//  what makes this an MPI + AMR problem and it is not implemented here. See
//  line_scan.hpp for the decomposition and the open questions that must be
//  settled before stage B can be written.
//
// What is here is still needed in the final algorithm, as two separate pieces:
//   - the reversed/forward within-block half-sums below are the second term of
//   the
//     combine step (stage C);
//   - the forward pass over the whole line inside each kernel is the block's
//   total along
//     that line, which is the payload that stage B must scan across blocks
//     (stage A).
//
// Three kernels are used, one per axis, because the memory layout makes them
// different problems rather than the same problem three times:
//
//  - x1 (along i): the scan axis is the *contiguous* index of the LayoutRight
//  fields, so
//    threads within a warp hold consecutive i. A cross-thread
//    Kokkos::parallel_scan over TeamThreadRange is both coalesced and parallel.
//  - x2, x3 (along j, k): the scan axis is a strided index. A cross-thread scan
//  here
//    would make each warp access nx1 elements apart, losing most of the
//    bandwidth. Instead each thread owns one whole line and walks it serially,
//    with the *thread* index still running over i. Access stays coalesced and
//    the scan itself is free.
//
// Both directions on an axis come from two passes over the line (forward, then
// reversed). Each pass reads the input once and writes the output once, so the
// whole operation is O(nmb * nvar * nkji) with a small constant.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"

void

    //----------------------------------------------------------------------------------------
    // x1 half-lines: exclusive scan along i, across the threads of a team.
    // One team per (m, n, k, j) line; the team's threads split the i range.

    void LineScanX1(int nmb, int nvar, int nvar_offset,
                    const RegionIndcs& indcs, const DvceArray5D<Real>& q,
                    const DvceArray6D<Real>& out) {
  const int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;

  // 4D outer loop: teams over (m, n, k, j), scan over i inside
  par_for_outer(
      "line_scan_x1", DevExeSpace(), 0, 0, 0, (nmb - 1), 0, (nvar - 1), ks, ke,
      js, je,
      KOKKOS_LAMBDA(TeamMember_t t, const int m, const int n, const int k,
                    const int j) {
        const int nv = nvar_offset + n;
        // MINUS_X1: exclusive prefix sum from the low face up to (not
        // including) i
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, is, ie + 1),
                              [=](const int i, Real& update, const bool final) {
                                const Real x = q(m, n, k, j, i);
                                if (final) {
                                  out(m, nv, MINUS_X1, k, j, i) = update;
                                }
                                update += x;
                              });

        // PLUS_X1: same scan with the index reversed, giving the suffix sum to
        // the high face
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, nx1),
                              [=](const int p, Real& update, const bool final) {
                                const int i = ie - p;
                                const Real x = q(m, n, k, j, i);
                                if (final) {
                                  out(m, nv, PLUS_X1, k, j, i) = update;
                                }
                                update += x;
                              });
      });
}

//----------------------------------------------------------------------------------------
// x2 half-lines: exclusive scan along j, serial within each thread.
// One thread per (m, n, k, i) line; the thread walks its own line over j.

void LineScanX2(int nmb, int nvar, int nvar_offset, const RegionIndcs& indcs,
                const DvceArray5D<Real>& q, const DvceArray6D<Real>& out) {
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;

  // 4D par_for decomposes the flat index as (n,k,j,i) with i fastest; the
  // ranges are chosen so those map to (m, n_var, k, i) and neighbouring threads
  // hold neighbouring i
  par_for(
      "line_scan_x2", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), ks, ke, is,
      ie, KOKKOS_LAMBDA(const int m, const int n, const int k, const int i) {
        const int nv = nvar_offset + n;
        Real run = 0.0;
        for (int j = js; j <= je; ++j) {
          out(m, nv, MINUS_X2, k, j, i) = run;
          run += q(m, n, k, j, i);
        }
        run = 0.0;
        for (int j = je; j >= js; --j) {
          out(m, nv, PLUS_X2, k, j, i) = run;
          run += q(m, n, k, j, i);
        }
      });
}

//----------------------------------------------------------------------------------------
// x3 half-lines: exclusive scan along k, serial within each thread.
// One thread per (m, n, j, i) line; the thread walks its own line over k.

void LineScanX3(int nmb, int nvar, int nvar_offset, const RegionIndcs& indcs,
                const DvceArray5D<Real>& q, const DvceArray6D<Real>& out) {
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;

  // 4D par_for decomposes the flat index as (n,k,j,i) with i fastest; the
  // ranges are chosen so those map to (m, n_var, j, i) and neighbouring threads
  // hold neighbouring i
  par_for(
      "line_scan_x3", DevExeSpace(), 0, (nmb - 1), 0, (nvar - 1), js, je, is,
      ie, KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
        const int nv = nvar_offset + n;
        Real run = 0.0;
        for (int k = ks; k <= ke; ++k) {
          out(m, nv, MINUS_X3, k, j, i) = run;
          run += q(m, n, k, j, i);
        }
        run = 0.0;
        for (int k = ke; k >= ks; --k) {
          out(m, nv, PLUS_X3, k, j, i) = run;
          run += q(m, n, k, j, i);
        }
      });
}

//----------------------------------------------------------------------------------------
//! \brief void LineScanWithinBlock(...)
//  \brief compute the WITHIN-BLOCK half-line exclusive prefix sums of every
//  variable.
//
// Output variable index is flattened across `vars` in order; `out` must be
// dimensioned (nmb, nvar_total, 6, nk, nj, ni).
//
// INCOMPLETE: sums stop at the MeshBlock boundary. See the file comment.

void LineScanWithinBlock(MeshBlockPack* ppack,
                         const std::vector<DvceArray5D<Real>>& vars,
                         DvceArray6D<Real> out) {
  auto& indcs = ppack->pmesh->mb_indcs;
  // nmb_thispack, *not* extent(0): physics arrays are allocated to
  // max(nmb_thispack, nmb_maxperrank) so the trailing slots are uninitialized.
  const int nmb = ppack->nmb_thispack;

  int nvar_offset = 0;
  for (auto& q : vars) {
    const int nvar = q.extent(1);
    LineScanX1(nmb, nvar, nvar_offset, indcs, q, out);
    LineScanX2(nmb, nvar, nvar_offset, indcs, q, out);
    LineScanX3(nmb, nvar, nvar_offset, indcs, q, out);
    nvar_offset += nvar;
  }
  return;
}
