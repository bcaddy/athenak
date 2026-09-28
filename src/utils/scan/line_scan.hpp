//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.hpp
//  \brief exclusive prefix sums of cell-centered fields along the six
//  half-lines of
//      each MeshBlock.
//
// For every active cell and every requested variable, computes six values: the
// sum of the variable over the cells strictly between this cell and each of the
// six faces of the MeshBlock, measured along a single row/column of cells.
//
//   out(m,n,PLUS_X1,k,j,i)  = sum_{i'=i+1}^{ie} q(m,n,k,j,i')
//   out(m,n,MINUS_X1,k,j,i) = sum_{i'=is}^{i-1} q(m,n,k,j,i')
//   ... and likewise for x2 (j) and x3 (k).
//
// Sums are exclusive: the cell's own value is never included. Each line is
// truncated at the MeshBlock boundary -- ghost cells are excluded and lines are
// NOT continued into neighboring MeshBlocks, so the whole operation is local to
// a rank and needs no communication.
#ifndef UTILS_LINE_SCAN_HPP_
#define UTILS_LINE_SCAN_HPP_

#include <vector>

#include "athena.hpp"

class MeshBlockPack;

//! the six half-line directions; index 3rd dimension of the output array
enum LineScanDir : int {
  PLUS_X1 = 0,
  MINUS_X1 = 1,
  PLUS_X2 = 2,
  MINUS_X2 = 3,
  PLUS_X3 = 4,
  MINUS_X3 = 5
};
constexpr int nline_scan_dir = 6;

//! \brief compute the six half-line exclusive prefix sums of each variable in
//!    `vars`, storing the result in `out` (dims m, nvar, 6, k, j, i).
//!
//! `vars` is a list of primitive (or conserved) field arrays to scan; all must
//! share the MeshBlock index range of `ppack` and have the same number of
//! variables. The output variable index is flattened across `vars` in order,
//! i.e. if `vars[0]` has 5 variables then out indices 0..4 refer to `vars[0]`
//! and 5.. refer to `vars[1]`.
void LineScan(MeshBlockPack* ppack, const std::vector<DvceArray5D<Real>>& vars,
              DvceArray6D<Real> out);

#endif  // UTILS_LINE_SCAN_HPP_
