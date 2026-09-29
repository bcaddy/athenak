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
#ifndef UTILS_SCAN_LINE_SCAN_HPP_
#define UTILS_SCAN_LINE_SCAN_HPP_

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

//----------------------------------------------------------------------------------------
//! \class LineScan
//! \brief Container for a single cell-centered field, shaped like the (nmb, k,
//! j, i) slice of Hydro/MHD u0 and w0 but with no variable axis.
//!
//! Allocated once in the constructor and never resized, for the same reason
//! Hydro/MHD allocate u0/w0 only in their constructor: the MeshBlock axis is
//! sized to max(nmb_thispack, nmb_maxperrank), which is an upper bound for the
//! whole run, so AMR refines into pre-allocated slots instead of growing the
//! View.  See line_scan.cpp.
//!
//! Two consequences to be aware of:
//!  - Only ACTIVE cells are allocated (nx1 x nx2 x nx3, no ghost zones), so
//!  this field is
//!    not boundary-communicated and cannot be used with a stencil that reads
//!    ghosts.
//!  - Kernels must iterate m over pmy_pack->nmb_thispack, never over
//!  extent_int(0).  The
//!    slots between those two are allocated for AMR headroom but hold no
//!    MeshBlock, and their contents are undefined once refinement has moved
//!    blocks around.

class LineScan {
 public:
  explicit LineScan(MeshBlockPack* ppack);
  ~LineScan();

  MeshBlockPack* pmy_pack;
  DvceArray4D<Real> data;  // (nmb_maxperrank, nk, nj, ni), active cells only

  void FillAll(Real value);  // write a constant into every element of data
};

#endif  // UTILS_SCAN_LINE_SCAN_HPP_
