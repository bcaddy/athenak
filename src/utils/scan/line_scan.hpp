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

class MeshBlockPack;

namespace line_scan {
//----------------------------------------------------------------------------------------
/*!
 * \class LineScan
 * \brief Perform either a prefix or suffix sum along a given axis.
 *
 */
class LineScan {
 public:
  explicit LineScan(MeshBlockPack* ppack);
  ~LineScan() = default;

  MeshBlockPack* pmy_pack;
  DvceArray4D<Real> scan_data;
};
}  // namespace line_scan
#endif  // UTILS_SCAN_LINE_SCAN_HPP_
