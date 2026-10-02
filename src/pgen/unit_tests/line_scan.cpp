//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.cpp
//  \brief Problem generator for unit tests of the block local line scans in
//  utils/scan/line_scan.hpp

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "utils/scan/line_scan.hpp"

namespace line_scan_test {
using line_scan::Direction;
using line_scan::ScanKind;

// Exactly representable values, so the scans can be compared exactly
KOKKOS_INLINE_FUNCTION Real CellValue(int m, int k, int j, int i) {
  return 1.0 + i + 7.0 * j + 13.0 * k + 101.0 * m;
}

// Ghost cells hold a huge value, so a scan that reads them fails
constexpr Real kGhostValue = 1.0e30;

//----------------------------------------------------------------------------------------
//! \fn int CheckScan()
//! \brief Runs one LineScan with a value_func that reads q, and compares every entry of
//! scan_data, including the line totals in the ghost cells, to a host reference.
//! Returns the number of mismatches.

template <Direction Dir, ScanKind Kind>
int CheckScan(MeshBlockPack *pmbp, const DvceArray4D<Real> &q) {
  auto value_func = KOKKOS_LAMBDA(const int m, const int k, const int j,
                                  const int i) -> Real {
    return q(m, k, j, i);
  };
  line_scan::LineScan<Dir, Kind, decltype(value_func)> scan(pmbp, value_func);
  // Fill with a sentinel so entries the scan never writes are caught
  Kokkos::deep_copy(scan.scan_data, -1.0);
  scan.BlockLocalScan();

  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scan.scan_data);
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int n = (Dir == Direction::I) ? indcs.nx1
              : (Dir == Direction::J) ? indcs.nx2 : indcs.nx3;
  const int di = (Dir == Direction::I), dj = (Dir == Direction::J),
            dk = (Dir == Direction::K);

  int nerr = 0;
  // Loop over the first real cell of each line in scan_data indices
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int k = scan.ks; k <= (dk ? scan.ks : scan.ke); ++k) {
      for (int j = scan.js; j <= (dj ? scan.js : scan.je); ++j) {
        for (int i = scan.is; i <= (di ? scan.is : scan.ie); ++i) {
          Real expected = 0.0;
          // s == n is the ghost cell holding the line total
          for (int s = 0; s <= n; ++s) {
            const int p = (Kind == ScanKind::Suffix) ? n - 1 - s : s;
            const int kp = k + dk * p, jp = j + dj * p, ip = i + di * p;
            if (h(m, kp, jp, ip) != expected) {
              if (nerr < 10) {
                std::cout << "  mismatch at m=" << m << " k=" << kp << " j=" << jp
                          << " i=" << ip << ": got " << h(m, kp, jp, ip)
                          << ", expected " << expected << std::endl;
              }
              ++nerr;
            }
            if (s < n) {
              expected += CellValue(m, kp - scan.ks + indcs.ks, jp - scan.js + indcs.js,
                                    ip - scan.is + indcs.is);
            }
          }
        }
      }
    }
  }

  const char *dir_name = (Dir == Direction::I) ? "I" :
                         (Dir == Direction::J) ? "J" : "K";
  const char *kind_name = (Kind == ScanKind::Prefix) ? "prefix" : "suffix";
  std::cout << "LineScan " << dir_name << " " << kind_name << ": "
            << ((nerr == 0) ? "passed" : "FAILED") << " (" << nerr << " mismatches)"
            << std::endl;
  return nerr;
}
}  // namespace line_scan_test

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::LineScan()
//! \brief Problem generator for unit tests of the block local line scans. Runs prefix
//! and suffix sums in every direction and exits with EXIT_FAILURE on any mismatch.

void ProblemGenerator::LineScan(ParameterInput *pin, const bool restart) {
  using line_scan::Direction;
  using line_scan::ScanKind;
  using line_scan_test::CheckScan;
  using line_scan_test::CellValue;
  using line_scan_test::kGhostValue;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int nmb = pmbp->nmb_thispack;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nc1 = indcs.nx1 + 2 * indcs.ng;
  const int nc2 = (indcs.nx2 > 1) ? indcs.nx2 + 2 * indcs.ng : 1;
  const int nc3 = (indcs.nx3 > 1) ? indcs.nx3 + 2 * indcs.ng : 1;

  // Field read by value_func, indexed like u0 with ghost cells
  DvceArray4D<Real> q("line_scan_test_q", nmb, nc3, nc2, nc1);
  par_for("line_scan_test_fill", DevExeSpace(), 0, nmb - 1, 0, nc3 - 1, 0, nc2 - 1,
          0, nc1 - 1, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const bool real = (i >= is && i <= ie && j >= js && j <= je && k >= ks &&
                               k <= ke);
            q(m, k, j, i) = real ? CellValue(m, k, j, i) : kGhostValue;
          });

  int nerr = 0;
  nerr += CheckScan<Direction::I, ScanKind::Prefix>(pmbp, q);
  nerr += CheckScan<Direction::I, ScanKind::Suffix>(pmbp, q);
  nerr += CheckScan<Direction::J, ScanKind::Prefix>(pmbp, q);
  nerr += CheckScan<Direction::J, ScanKind::Suffix>(pmbp, q);
  nerr += CheckScan<Direction::K, ScanKind::Prefix>(pmbp, q);
  nerr += CheckScan<Direction::K, ScanKind::Suffix>(pmbp, q);

  if (nerr != 0) {
    std::cout << "LineScan unit test failed with " << nerr << " mismatches" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "LineScan unit test passed" << std::endl;
}
