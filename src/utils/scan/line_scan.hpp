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
  DvceArray4D<Real> scan_data;
};
}  // namespace line_scan
#endif  // UTILS_SCAN_LINE_SCAN_HPP_
