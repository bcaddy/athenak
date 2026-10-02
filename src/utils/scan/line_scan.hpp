//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.hpp
//  \brief Header file for exclusive prefix scans along any cartisian direction

#ifndef UTILS_SCAN_LINE_SCAN_HPP_
#define UTILS_SCAN_LINE_SCAN_HPP_

#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"

namespace line_scan {
// Scan direction and kind. Outside of LineScan so they can be named without the
// class template arguments
enum class Direction { I, J, K };
enum class ScanKind { Prefix, Suffix };

//----------------------------------------------------------------------------------------
/*!
 * \class LineScan
 * \brief Perform either an exclusive prefix or exclusive suffix scan along a
 * given axis
 *
 * Since Dir and Kind are given explicitly, ValueFunc must be too, e.g. with
 * decltype:
 * \code
 * auto f = KOKKOS_LAMBDA(const int m, const int k, const int j, const int i)
 *     -> Real { return u0(m, IDN, k, j, i); };
 * line_scan::LineScan<line_scan::Direction::I, line_scan::ScanKind::Prefix,
 *                     decltype(f)> scan(pmbp, f);
 * \endcode
 *
 * \tparam Dir The direction along which the scan is performed.
 * \tparam Kind Whether the scan is an exclusive prefix or suffix scan.
 * \tparam ValueFunc Type of a device callable that returns the value to scan at
 * a cell. A KOKKOS_LAMBDA must have the signature
 * \code
 * KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) -> Real
 * \endcode
 * where m is the meshblock and k, j, i are MeshBlock (mb_indcs) indices, which
 * include ghost cells.
 */
template <Direction Dir, ScanKind Kind, typename ValueFunc>
class LineScan {
 public:
  static constexpr Direction direction = Dir;
  static constexpr ScanKind scan_kind = Kind;

  /*!
   * \param value_func The callable that computes the value to scan.
   */
  LineScan(MeshBlockPack* ppack, const ValueFunc& value_func)
      : pmy_pack(ppack),
        value_func(value_func),
        nmb(std::max(pmy_pack->nmb_thispack, pmy_pack->pmesh->nmb_maxperrank)),
        // Number of real cells plus 2 in the scan direction to store the block
        // wide scan
        ni(ppack->pmesh->mb_indcs.nx1 + ((Dir == Direction::I) ? 2 : 0)),
        nj(ppack->pmesh->mb_indcs.nx2 + ((Dir == Direction::J) ? 2 : 0)),
        nk(ppack->pmesh->mb_indcs.nx3 + ((Dir == Direction::K) ? 2 : 0)),
        // Real cells start at 1 in the scan direction (index 0 and n-1 hold the
        // block wide scan) and at 0 otherwise
        is((Dir == Direction::I) ? 1 : 0),
        ie(is + ppack->pmesh->mb_indcs.nx1 - 1),
        js((Dir == Direction::J) ? 1 : 0),
        je(js + ppack->pmesh->mb_indcs.nx2 - 1),
        ks((Dir == Direction::K) ? 1 : 0),
        ke(ks + ppack->pmesh->mb_indcs.nx3 - 1),
        // Allocate storage
        scan_data("scan_data", nmb, nk, nj, ni) {}

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
   * \brief Run the block local scan in the chosen direction.
   */
  void BlockLocalScan() {
    if constexpr (Dir == Direction::I) {
      BlockLocalScan_I();
    } else {
      BlockLocalScan_JK();
    }
  }

  /*!
   * \brief Exclusive prefix or suffix scan along i of value_func over the real
   * cells of each meshblock, stored in scan_data. The line total is stored in
   * the ghost cell past the end of the scan: ie+1 for a prefix scan, is-1 for a
   * suffix scan.
   *
   * Each (m, k, j) line is handled by one team thread, with the thread's vector
   * lanes splitting the line. This suits short lines (nx1 <= 32), where a whole
   * team per line would leave most of its threads idle.
   */
  void BlockLocalScan_I() {
    // Local copies so the device lambda doesn't capture the host `this` pointer
    auto scan_data_ = scan_data;
    auto value_func_ = value_func;
    const int is_ = is, ie_ = ie, nj_ = nj, nk_ = nk;
    const int nx = ie - is + 1;
    // Only the meshblocks in use; scan_data is sized for the maximum, nmb
    const int nlines = pmy_pack->nmb_thispack * nk * nj;

    // Offsets from scan_data indices to MeshBlock indices, which include ghosts
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const int ioff = indcs.is - is, joff = indcs.js - js, koff = indcs.ks - ks;

    // Vector length: the smallest power of two >= nx (Kokkos rounds down to a
    // power of two), capped at the CUDA warp size and the backend maximum
    const int vlen =
        std::min({static_cast<int>(Kokkos::bit_ceil(static_cast<unsigned>(nx))),
                  32, Kokkos::TeamPolicy<>::vector_length_max()});

    // About 128 threads per team. Host backends allow at most concurrency()
    // threads per team (1 for Serial)
    const int team_size = std::min(128 / vlen, DevExeSpace().concurrency());
    const int n_league = (nlines + team_size - 1) / team_size;

    // Perform the scan
    Kokkos::parallel_for(
        "BlockLocalScan_I",
        Kokkos::TeamPolicy<>(DevExeSpace(), n_league, team_size, vlen),
        KOKKOS_LAMBDA(TeamMember_t t) {
          const int line = t.league_rank() * t.team_size() + t.team_rank();
          if (line >= nlines) return;
          const int m = line / (nk_ * nj_);
          const int k = (line / nj_) % nk_;
          const int j = line % nj_;

          // value_func is called inside the scan, so it may run more than once
          // per cell; profiling showed this is faster than evaluating it once
          // into scan_data first for typical value_funcs. The range starts at 0
          // since the CUDA vector scan ignores a nonzero start
          Real total;
          Kokkos::parallel_scan(
              Kokkos::ThreadVectorRange(t, nx),
              [=](const int p, Real& update, const bool final) {
                const int i = (Kind == ScanKind::Suffix) ? ie_ - p : is_ + p;
                const Real x = value_func_(m, k + koff, j + joff, i + ioff);
                if (final) {
                  scan_data_(m, k, j, i) = update;
                }
                update += x;
              },
              total);
          Kokkos::single(Kokkos::PerThread(t), [=]() {
            scan_data_(m, k, j,
                       (Kind == ScanKind::Suffix) ? is_ - 1 : ie_ + 1) = total;
          });
        });
  }

  /*!
   * \brief Exclusive prefix or suffix scan along j or k of value_func over the
   * real cells of each meshblock, stored in scan_data. The line total is stored
   * in the ghost cell past the end of the scan, as in BlockLocalScan_I.
   *
   * Each line is handled by one thread that steps along it serially, so
   * value_func is called once per cell. Neighbouring threads handle
   * neighbouring i, so their loads and stores are coalesced. (This layout is
   * 3-9x slower than BlockLocalScan_I for scans along i, where neighbouring
   * threads would be a whole line apart.)
   */
  void BlockLocalScan_JK() {
    // Local copies so the device lambda doesn't capture the host `this` pointer
    auto scan_data_ = scan_data;
    auto value_func_ = value_func;
    // Number of real cells along the line
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const int n = (Dir == Direction::J) ? indcs.nx2 : indcs.nx3;

    // Offsets from scan_data indices to MeshBlock indices, which include ghosts
    const int ioff = indcs.is - is, joff = indcs.js - js, koff = indcs.ks - ks;

    // One thread per line: the scan direction's range is just its first real
    // cell
    par_for(
        "BlockLocalScan_JK", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1, ks,
        (Dir == Direction::K) ? ks : ke, js, (Dir == Direction::J) ? js : je,
        is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real sum = 0.0;
          // s == n writes the line total to the ghost cell past the end
          for (int s = 0; s <= n; ++s) {
            // Offset from the first real cell, walking backwards for a
            // suffix scan
            const int p = (Kind == ScanKind::Suffix) ? n - 1 - s : s;
            const int kp = (Dir == Direction::K) ? k + p : k;
            const int jp = (Dir == Direction::J) ? j + p : j;
            scan_data_(m, kp, jp, i) = sum;
            if (s < n) {
              sum += value_func_(m, kp + koff, jp + joff, i + ioff);
            }
          }
        });
  }
};
}  // namespace line_scan
#endif  // UTILS_SCAN_LINE_SCAN_HPP_
