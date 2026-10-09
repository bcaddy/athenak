//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.hpp
//  \brief Header file for exclusive prefix and suffix sums along any cartesian
//  direction

#ifndef UTILS_SCAN_LINE_SCAN_HPP_
#define UTILS_SCAN_LINE_SCAN_HPP_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif  // MPI_PARALLEL_ENABLED

namespace line_scan {
// Outside of LineScan so they can be named without the template arguments
enum class Direction { I, J, K };
enum class ScanKind { Prefix, Suffix };
// The work a LineScan currently has in flight
enum class Stage {
  NotStarted,
  BlockLocalScan,
  Exchange,
  AddUpstream,
  Completed
};

// An upstream MeshBlock whose line totals add to a MeshBlock's upstream sum
struct Source {
  int face;    // index in faces (local source) or recvbuf (remote source)
  int dlevel;  // source level minus target level
  // Offset of the finer face's origin from the coarser one's, in finer cells
  int off1, off2;
};

// The faces in sendbuf or recvbuf exchanged with one rank
struct Peer {
  int rank;
  int first_face;
  int nfaces;
};

//----------------------------------------------------------------------------------------
/*!
 * \class LineScan
 * \brief Compute either an exclusive prefix sum or exclusive suffix sum along a
 * given axis across the entire domain
 *
 * \details Each use writes the global scan to scan_data: each MeshBlock's
 * block local scan plus the sum of the line totals of all the MeshBlocks
 * upstream of it, on any rank. The domain boundaries, periodic or not, start
 * the scan at zero.
 *
 * With mesh refinement, upstream totals are mapped to the resolution of the
 * target MeshBlock: finer ones are area averaged and coarser ones linearly
 * prolongated with minmod slopes (one sided at face edges, scaled to stay
 * non-negative).
 *
 * Each LineScan runs its kernels on its own execution space instance and has
 * its own MPI communicator, so several can run concurrently. Starting a scan
 * fences all prior device work, so its inputs are ready and earlier reads of
 * scan_data are done; scan_data is ready once Driver returns Completed. The
 * kernel methods are public only because nvcc requires that of
 * methods containing a `KOKKOS_LAMBDA`; their Impl prefix marks them as
 * internal.
 *
 * Requirements on the caller:
 *  - All GPU operations that `value_func` depends on must be complete before
 * starting a scan. Easiest done by calling a Kokkos Fence before running scans
 *  - Every rank constructs LineScans in the same order, since the constructor
 *    duplicates a communicator, which is collective.
 *  - Started scans are driven to completion on every rank, polled round-robin
 *    or completed in the same order everywhere, or ranks can deadlock.
 *  - No scan is in flight when the mesh is refined.
 *  - LineScans are destroyed before MPI and Kokkos are finalized.
 *  - LineScans are destroyed in the same order on every rank, since the
 *    destructor frees a communicator, which is collective.
 *  - value_func only captures data that persists across mesh refinement, e.g.
 *    u0 or w0; MeshBlock geometry is passed to it instead. With mesh
 *    refinement the summed value must be extensive (e.g. density times dx)
 *    and non-negative.
 *
 * Since Dir and Kind are given explicitly, ValueFunc must be too, e.g. with
 * decltype:
 * \code
 * auto f = KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
 *                        const RegionSize& size) -> Real {
 *   return w0(m, IDN, k, j, i) * size.dx1;
 * };
 * line_scan::LineScan<line_scan::Direction::I, line_scan::ScanKind::Prefix,
 *                     decltype(f)> scan(pmbp, f);
 * \endcode
 *
 * \tparam Dir The direction along which the scan is performed.
 * \tparam Kind Whether to compute an exclusive prefix or suffix sum.
 * \tparam ValueFunc Type of a device callable returning the value to sum at a
 * cell, with the signature
 * \code
 * KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
 *               const RegionSize& size) -> Real
 * \endcode
 * where m is the MeshBlock's index in the MeshBlockPack, k, j, i are its cell
 * indices (mb_indcs, including ghost cells), and size is its mb_size entry.
 *
 * \note Only sums are supported. Another associative operation would need its
 * identity in place of 0, a Kokkos reducer in ImplBlockLocalScan_I, and its own
 * restriction and prolongation in the exchange.
 */
template <Direction Dir, ScanKind Kind, typename ValueFunc>
class LineScan {
 public:
  /*!
   * \brief Allocate the scan's arrays and duplicate MPI_COMM_WORLD, which is
   * collective over all ranks.
   * \param ppack The MeshBlockPack to scan over.
   * \param value_func Returns the value to sum at each cell.
   */
  LineScan(MeshBlockPack* ppack, const ValueFunc& value_func)
      // Not initialized, since every scan writes all the entries it uses
      : scan_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "scan_data"),
                  std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank),
                  ppack->pmesh->mb_indcs.nx3, ppack->pmesh->mb_indcs.nx2,
                  ppack->pmesh->mb_indcs.nx1),
        pmy_pack(ppack),
        value_func(value_func),
        exec_space(Kokkos::Experimental::partition_space(DevExeSpace(), 1)[0]),
        nmb(scan_data.extent_int(0)),
        n_scan(SelectWithDirection(Dir, ppack->pmesh->mb_indcs.nx1,
                                   ppack->pmesh->mb_indcs.nx2,
                                   ppack->pmesh->mb_indcs.nx3)),
        n1(SelectWithDirection(t1_dir, ppack->pmesh->mb_indcs.nx1,
                               ppack->pmesh->mb_indcs.nx2,
                               ppack->pmesh->mb_indcs.nx3)),
        n2(SelectWithDirection(t2_dir, ppack->pmesh->mb_indcs.nx1,
                               ppack->pmesh->mb_indcs.nx2,
                               ppack->pmesh->mb_indcs.nx3)),
        faces(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "LineScan::faces"),
            nmb, n2, n1),
        upstream_sum(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        "LineScan::upstream_sum"),
                     nmb, n2, n1),
        local_start(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                       "LineScan::local_start"),
                    nmb + 1),
        remote_start(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        "LineScan::remote_start"),
                     nmb + 1),
        h_local_start("LineScan::h_local_start", nmb + 1),
        h_remote_start("LineScan::h_remote_start", nmb + 1) {
#if MPI_PARALLEL_ENABLED
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
#endif  // MPI_PARALLEL_ENABLED
  }

  /*!
   * \brief Not copyable, since each LineScan owns its MPI communicator.
   */
  LineScan(const LineScan&) = delete;
  LineScan& operator=(const LineScan&) = delete;

  /*!
   * \brief Free the MPI communicator. Fails if a scan is in flight.
   */
  ~LineScan() {
    if (stage != Stage::NotStarted && stage != Stage::Completed) {
      FatalError("LineScan destroyed while a scan is in flight");
    }
#if MPI_PARALLEL_ENABLED
    MPI_Comm_free(&comm);
#endif  // MPI_PARALLEL_ENABLED
  }

  /*!
   * \brief Advance the scan. Only starting a scan blocks, to fence all prior
   * device work.
   * \return The current stage. Once Completed, scan_data can be read without
   * a fence.
   */
  Stage Driver() {
    switch (stage) {
      // Build the exchange plan and run the block local scans
      case Stage::NotStarted:
        // Rebuild the plan on the first run and after the mesh changes
        if (plan_mesh_seq != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
          BuildExchangePlan();
        }

        // If MPI is enabled then post the receives now
        PostReceives();

        // Choose the correct block local scan for the direction
        if constexpr (Dir == Direction::I) {
          ImplBlockLocalScan_I();
        } else {
          ImplBlockLocalScan_JK();
        }

        // If there are meshblock faces to send over MPI then queue the kernel
        // that packs the MPI buffers.
        if (nsend_faces > 0) {
          ImplPack();
        } else {
          // If no MPI is needed then start accumulating local values into
          // upstream_sum. If MPI is needed then this is delayed so the MPI can
          // start as soon as possible. Otherwise the check for if this GPU
          // stream is idle would require that both this and the packing kernel
          // are done before MPI is started
          ImplAccumulate(/*remote=*/false);
        }

        // Update the stage and break
        stage = Stage::BlockLocalScan;
        break;
      case Stage::BlockLocalScan:
#if MPI_PARALLEL_ENABLED
        // Check if any MPI sends/recvs are required. If not then mark the
        // exchange as done since ImplAccumulate would have already been queued
        // by the previous stage
        if (nsend_faces > 0) {
          // Check if the send buffer is packed yet. If not then immediately
          // return
          if (!ExecSpaceIdle(exec_space)) break;

          // If MPI is enabled then post the sends now
          PostSends();

          // Run the local accumulations while MPI messages are in flight
          ImplAccumulate(/*remote=*/false);
        }
#endif  // MPI_PARALLEL_ENABLED
        stage = Stage::Exchange;
        break;
      case Stage::Exchange:
#if MPI_PARALLEL_ENABLED
        //  Check if the MPI receives are done, if not then return
        if (!TestMPIStatus(recv_reqs)) break;
        if (nrecv_faces > 0) {
          // Now that the MPI receives are done, add all the data received into
          // the local accumulations
          ImplAccumulate(/*remote=*/true);
        }
#endif  // MPI_PARALLEL_ENABLED
        // Add all the accumulated scan data to each cell's value.
        ImplAddUpstream();
        stage = Stage::AddUpstream;
        break;
      case Stage::AddUpstream:
        // This stage verifies that everything has completed. At this point all
        // MPI receives are complete so it just checks if the sends are
        // completes and if the GPU stream is idle, i.e. all the kernels have
        // completed
#if MPI_PARALLEL_ENABLED
        if (!TestMPIStatus(send_reqs)) {
          break;
        }
#endif  // MPI_PARALLEL_ENABLED
        if (ExecSpaceIdle(exec_space)) {
          stage = Stage::Completed;
        }
        break;
      case Stage::Completed:
        break;
    }
    return stage;
  }

  /*!
   * \brief Make a Completed scan ready to run again. Fails if a scan is in
   * flight.
   */
  void Reset() {
    if (stage != Stage::Completed && stage != Stage::NotStarted) {
      FatalError("LineScan::Reset called while a scan is in flight");
    }
    stage = Stage::NotStarted;
  }

  // The global exclusive scan, indexed (m, k, j, i) over each MeshBlock's real
  // cells from 0
  const DvceArray4D<Real> scan_data;

  //--------------------------------------------------------------------------------------
  // Kernels. Public only because nvcc requires that of methods containing a
  // KOKKOS_LAMBDA; the Impl prefix marks them as not for use outside LineScan

  /*!
   * \brief Exclusive scan along i of value_func over each MeshBlock's real
   * cells, with the line totals stored in faces.
   *
   * Each (m, k, j) line is one team thread, whose vector lanes split the line.
   * This suits short lines (nx1 <= 32), where a whole team per line would
   * leave most of its threads idle.
   */
  void ImplBlockLocalScan_I() {
    // Local copies so the device lambda doesn't capture `this`
    auto scan_data_ = scan_data;
    auto faces_ = faces;
    auto value_func_ = value_func;
    // Recreated by mesh refinement, so read it at each launch
    auto mb_size = pmy_pack->pmb->mb_size.d_view;
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const int nx = n_scan, nj = indcs.nx2, nk = indcs.nx3;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int nlines = pmy_pack->nmb_thispack * nk * nj;

    // The smallest power of two >= nx (Kokkos rounds others down), capped at
    // the backend maximum (the warp or wavefront size on GPUs)
    const int vlen =
        std::min(static_cast<int>(Kokkos::bit_ceil(static_cast<unsigned>(nx))),
                 Kokkos::TeamPolicy<>::vector_length_max());
    // About 128 threads per team, at most concurrency() on host backends, and
    // at least 1 since host vectors can be long
    const int team_size =
        std::max(1, std::min(128 / vlen, exec_space.concurrency()));
    const int n_league = (nlines + team_size - 1) / team_size;

    Kokkos::parallel_for(
        "BlockLocalScan_I",
        Kokkos::TeamPolicy<>(exec_space, n_league, team_size, vlen),
        KOKKOS_LAMBDA(TeamMember_t t) {
          const int line = t.league_rank() * t.team_size() + t.team_rank();
          if (line >= nlines) return;
          const int m = line / (nk * nj);
          const int k = (line / nj) % nk;
          const int j = line % nj;
          const RegionSize size = mb_size(m);

          // value_func may run more than once per cell (profiled faster
          // than storing it). The range starts at 0 since the CUDA vector scan
          // ignores a nonzero start
          Real total;
          Kokkos::parallel_scan(
              Kokkos::ThreadVectorRange(t, nx),
              [=](const int p, Real& update, const bool final) {
                const int i = (Kind == ScanKind::Suffix) ? nx - 1 - p : p;
                const Real x = value_func_(m, k + ks, j + js, i + is, size);
                if (final) {
                  scan_data_(m, k, j, i) = update;
                }
                update += x;
              },
              total);
          Kokkos::single(Kokkos::PerThread(t),
                         [=]() { faces_(m, k, j) = total; });
        });
  }

  /*!
   * \brief Exclusive scan along j or k, stored as in ImplBlockLocalScan_I.
   *
   * Each line is one thread stepping along it, so value_func runs once per
   * cell and neighbouring threads (neighbouring i) access memory coalesced.
   * This is 3-9x slower than ImplBlockLocalScan_I for scans along i.
   */
  void ImplBlockLocalScan_JK() {
    auto scan_data_ = scan_data;
    auto faces_ = faces;
    auto value_func_ = value_func;
    auto mb_size = pmy_pack->pmb->mb_size.d_view;
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const int n = n_scan;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;

    // One thread per line, starting at its first cell
    par_for(
        "BlockLocalScan_JK", exec_space, 0, pmy_pack->nmb_thispack - 1, 0,
        (Dir == Direction::K) ? 0 : indcs.nx3 - 1, 0,
        (Dir == Direction::J) ? 0 : indcs.nx2 - 1, 0, indcs.nx1 - 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          const RegionSize size = mb_size(m);
          Real sum = 0.0;
          for (int s = 0; s < n; ++s) {
            const int p = (Kind == ScanKind::Suffix) ? n - 1 - s : s;
            const int kp = (Dir == Direction::K) ? p : k;
            const int jp = (Dir == Direction::J) ? p : j;
            scan_data_(m, kp, jp, i) = sum;
            sum += value_func_(m, kp + ks, jp + js, i + is, size);
          }
          // The face cell is (k, i) along j and (j, i) along k
          faces_(m, (Dir == Direction::J) ? k : j, i) = sum;
        });
  }

  /*!
   * \brief Copy the faces other ranks need into sendbuf.
   */
  void ImplPack() {
    auto faces_ = faces;
    auto sendbuf_ = sendbuf;
    auto send_m_ = send_m;
    par_for(
        "LineScan::Pack", exec_space, 0, nsend_faces - 1, 0, n2 - 1, 0, n1 - 1,
        KOKKOS_LAMBDA(const int f, const int t2, const int t1) {
          sendbuf_(f, t2, t1) = faces_(send_m_(f), t2, t1);
        });
  }

  /*!
   * \brief Sum each MeshBlock's local or remote sources into upstream_sum.
   * The local pass overwrites it, the remote pass adds to it. This all performs
   * the prolongation and restriction
   * \param remote Whether to sum the remote sources (in recvbuf) rather than
   * the local ones (in faces).
   */
  void ImplAccumulate(const bool remote) {
    auto upstream_sum_ = upstream_sum;
    auto start = remote ? remote_start : local_start;
    auto sources = remote ? remote_sources : local_sources;
    auto src_faces = remote ? recvbuf : faces;
    const int n1_ = n1, n2_ = n2;
    par_for(
        "LineScan::Accumulate", exec_space, 0, pmy_pack->nmb_thispack - 1, 0,
        n2 - 1, 0, n1 - 1,
        KOKKOS_LAMBDA(const int m, const int t2, const int t1) {
          Real sum = 0.0;
          for (int s = start(m); s < start(m + 1); ++s) {
            sum += MapToCell(sources(s), src_faces, t1, t2, n1_, n2_);
          }
          upstream_sum_(m, t2, t1) =
              remote ? upstream_sum_(m, t2, t1) + sum : sum;
        });
  }

  /*!
   * \brief Add each MeshBlock's upstream sum to its block local scan.
   */
  void ImplAddUpstream() {
    auto scan_data_ = scan_data;
    auto upstream_sum_ = upstream_sum;
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    par_for(
        "LineScan::AddUpstream", exec_space, 0, pmy_pack->nmb_thispack - 1, 0,
        indcs.nx3 - 1, 0, indcs.nx2 - 1, 0, indcs.nx1 - 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          // The face cell is (k, j) along i, (k, i) along j and (j, i) along k
          const int t1 = (Dir == Direction::I) ? j : i;
          const int t2 = (Dir == Direction::K) ? j : k;
          scan_data_(m, k, j, i) += upstream_sum_(m, t2, t1);
        });
  }

 private:
  // The two transverse directions, across the scan, in increasing order
  static constexpr Direction t1_dir =
      (Dir == Direction::I) ? Direction::J : Direction::I;
  static constexpr Direction t2_dir =
      (Dir == Direction::K) ? Direction::J : Direction::K;

  MeshBlockPack* pmy_pack;
  const ValueFunc value_func;
  // Declared before the Views it allocates
  const DevExeSpace exec_space;

  // MeshBlocks in scan_data
  const int nmb;
  // Real cells along the scan and the two transverse directions (1 if unused)
  const int n_scan, n1, n2;

  // Each MeshBlock's line totals, and the sum of its upstream MeshBlocks' line
  // totals. Indexed (m, t2, t1)
  const DvceArray3D<Real> faces, upstream_sum;

  Stage stage = Stage::NotStarted;

  //--------------------------------------------------------------------------------------
  // Exchange plan, built by Driver when the mesh changes

  // Sources of local MeshBlock m: sources(start(m)) to sources(start(m+1)-1),
  // split into those on this rank and those on others
  const DvceArray1D<int> local_start, remote_start;
  DvceArray1D<Source> local_sources, remote_sources;
  // Faces exchanged with other ranks, grouped by rank. send_m is the local
  // MeshBlock of each face in sendbuf
  DvceArray1D<int> send_m;
  DvceArray3D<Real> sendbuf, recvbuf;

  // Host copies of the plan, kept alive for the asynchronous copies. The
  // source and send_m arrays only grow, so only their first n_local_sources,
  // n_remote_sources and nsend_faces entries are in use
  const HostArray1D<int> h_local_start, h_remote_start;
  HostArray1D<Source> h_local_sources, h_remote_sources;
  HostArray1D<int> h_send_m;
  int n_local_sources = 0, n_remote_sources = 0;

  std::vector<Peer> send_peers, recv_peers;
  int nsend_faces = 0, nrecv_faces = 0;

  int plan_mesh_seq = -1;  // mesh update the plan was built for

#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  std::vector<MPI_Request> send_reqs, recv_reqs;
#endif  // MPI_PARALLEL_ENABLED

  //--------------------------------------------------------------------------------------
  // Mapping line totals between refinement levels

  /*!
   * \brief A source face's line totals mapped to target face cell (t1, t2).
   * \param src The source.
   * \param face The faces holding it: faces or recvbuf.
   * \param t1, t2 The target face cell.
   * \param n1, n2 Cells per face along t1 and t2.
   */
  KOKKOS_INLINE_FUNCTION
  static Real MapToCell(const Source& src, const DvceArray3D<Real>& face,
                        const int t1, const int t2, const int n1,
                        const int n2) {
    if (src.dlevel == 0) return face(src.face, t2, t1);
    if (src.dlevel > 0) return RestrictToCell(src, face, t1, t2, n1, n2);
    return ProlongToCell(src, face, t1, t2, n1, n2);
  }

  /*!
   * \brief Area average of the finer source cells inside target cell
   * (t1, t2), each covering 1/r of it per used dimension. Only cells in the
   * source face count, which matters when it is smaller than a target cell.
   * Arguments as in MapToCell.
   */
  KOKKOS_INLINE_FUNCTION
  static Real RestrictToCell(const Source& src, const DvceArray3D<Real>& face,
                             const int t1, const int t2, const int n1,
                             const int n2) {
    const int r = std::int64_t{1} << src.dlevel;
    const int r1 = (n1 > 1) ? r : 1, r2 = (n2 > 1) ? r : 1;
    const int lo1 = t1 * r1 - src.off1, lo2 = t2 * r2 - src.off2;
    const int hi1 = lo1 + r1, hi2 = lo2 + r2;
    const int c1_lo = Kokkos::max(lo1, 0);
    const int c1_hi = Kokkos::min(hi1, n1);
    const int c2_lo = Kokkos::max(lo2, 0);
    const int c2_hi = Kokkos::min(hi2, n2);
    Real sum = 0.0;
    for (std::int64_t c2 = c2_lo; c2 < c2_hi; ++c2) {
      for (std::int64_t c1 = c1_lo; c1 < c1_hi; ++c1) {
        sum += face(src.face, c2, c1);
      }
    }
    return sum / static_cast<Real>(r1 * r2);
  }

  /*!
   * \brief Linear prolongation of a coarser source face to the center of
   * target cell (t1, t2). Arguments as in MapToCell.
   */
  KOKKOS_INLINE_FUNCTION
  static Real ProlongToCell(const Source& src, const DvceArray3D<Real>& face,
                            const int t1, const int t2, const int n1,
                            const int n2) {
    const std::int64_t r = std::int64_t{1} << -src.dlevel;
    const std::int64_t r1 = (n1 > 1) ? r : 1, r2 = (n2 > 1) ? r : 1;
    // The source cell containing the target cell, and the target cell
    // center's position in it, from -1/2 to 1/2
    const std::int64_t p1 = src.off1 + t1, p2 = src.off2 + t2;
    const int c1 = static_cast<int>(p1 / r1), c2 = static_cast<int>(p2 / r2);
    const Real xi1 = (static_cast<Real>(p1 % r1) + 0.5) / r1 - 0.5;
    const Real xi2 = (static_cast<Real>(p2 % r2) + 0.5) / r2 - 0.5;

    const int f = src.face;
    const Real u = face(f, c2, c1);
    Real s1 = 0.0, s2 = 0.0;
    if (n1 > 1) {
      s1 = Slope(c1 > 0 ? u - face(f, c2, c1 - 1) : 0.0,
                 c1 < n1 - 1 ? face(f, c2, c1 + 1) - u : 0.0, c1 > 0,
                 c1 < n1 - 1);
    }
    if (n2 > 1) {
      s2 = Slope(c2 > 0 ? u - face(f, c2 - 1, c1) : 0.0,
                 c2 < n2 - 1 ? face(f, c2 + 1, c1) - u : 0.0, c2 > 0,
                 c2 < n2 - 1);
    }
    // Scale the slopes so the profile's minimum, u - (|s1| + |s2|) / 2, isn't
    // negative
    const Real slope_sum = fabs(s1) + fabs(s2);
    if (slope_sum > 2.0 * u) {
      const Real scale = fmax(2.0 * u, 0.0) / slope_sum;
      s1 *= scale;
      s2 *= scale;
    }
    return u + s1 * xi1 + s2 * xi2;
  }

  /*!
   * \brief Limited slope of a source face cell.
   * \param dl, dr Differences from the left and to the right neighbours.
   * \param has_left, has_right Whether those neighbours are in the face.
   * \return The minmod of dl and dr, or the one that exists at a face edge.
   */
  KOKKOS_INLINE_FUNCTION
  static Real Slope(const Real dl, const Real dr, const bool has_left,
                    const bool has_right) {
    if (has_left && has_right) {
      return 0.5 * (SIGN(dl) + SIGN(dr)) * fmin(fabs(dl), fabs(dr));
    }
    return has_left ? dl : dr;
  }

  //--------------------------------------------------------------------------------------
  // MPI communication, all non-blocking

  /*!
   * \brief Post a receive into recvbuf from each peer rank.
   */
  void PostReceives() {
#if MPI_PARALLEL_ENABLED
    const int face_size = n1 * n2;
    for (std::size_t p = 0; p < recv_peers.size(); ++p) {
      const Peer& peer = recv_peers[p];
      MPI_Irecv(recvbuf.data() + peer.first_face * face_size,
                peer.nfaces * face_size, MPI_ATHENA_REAL, peer.rank, 0, comm,
                &recv_reqs[p]);
    }
#endif  // MPI_PARALLEL_ENABLED
  }

  /*!
   * \brief Send the packed sendbuf to each peer rank.
   */
  void PostSends() {
#if MPI_PARALLEL_ENABLED
    const int face_size = n1 * n2;
    for (std::size_t p = 0; p < send_peers.size(); ++p) {
      const Peer& peer = send_peers[p];
      MPI_Isend(sendbuf.data() + peer.first_face * face_size,
                peer.nfaces * face_size, MPI_ATHENA_REAL, peer.rank, 0, comm,
                &send_reqs[p]);
    }
#endif  // MPI_PARALLEL_ENABLED
  }

#if MPI_PARALLEL_ENABLED
  /*!
   * \brief Whether all of reqs have completed, without blocking. Calling it
   * also lets MPI progress them.
   */
  static bool TestMPIStatus(std::vector<MPI_Request>& reqs) {
    int flag;
    MPI_Testall(static_cast<int>(reqs.size()), reqs.data(), &flag,
                MPI_STATUSES_IGNORE);
    return static_cast<bool>(flag);
  }
#endif  // MPI_PARALLEL_ENABLED

  //--------------------------------------------------------------------------------------
  // Building the exchange plan

  /*!
   * \brief Build the exchange plan for the current mesh and copy it to the
   * device, without fencing. Called only by Driver, on the first run and after
   * the mesh changes.
   */
  void BuildExchangePlan() {
    ComputePlan();

    // Make the device arrays big enough. Allocating synchronizes the whole
    // device, so this only allocates on the first run and when mesh refinement
    // needs more room than is currently allocated
    Reserve(local_sources, n_local_sources, "LineScan::local_sources");
    Reserve(remote_sources, n_remote_sources, "LineScan::remote_sources");
    Reserve(send_m, nsend_faces, "LineScan::send_m");
    Reserve(sendbuf, nsend_faces, "LineScan::sendbuf");
    Reserve(recvbuf, nrecv_faces, "LineScan::recvbuf");

    // Copy the plan arrays to the device
    Kokkos::deep_copy(exec_space, local_start, h_local_start);
    Kokkos::deep_copy(exec_space, remote_start, h_remote_start);
    Kokkos::deep_copy(exec_space, local_sources, h_local_sources);
    Kokkos::deep_copy(exec_space, remote_sources, h_remote_sources);
    Kokkos::deep_copy(exec_space, send_m, h_send_m);

    // One MPI request per peer
#if MPI_PARALLEL_ENABLED
    send_reqs.assign(send_peers.size(), MPI_REQUEST_NULL);
    recv_reqs.assign(recv_peers.size(), MPI_REQUEST_NULL);
#endif  // MPI_PARALLEL_ENABLED

    // Update the mesh configuration that this plan was built for
    plan_mesh_seq = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();
  }

  /*!
   * \brief Find, on the host, each local MeshBlock's upstream sources and the
   * faces to exchange with each rank, from the mesh layout every rank holds.
   * Both sides of a message derive the same faces, so no communication is
   * needed.
   */
  void ComputePlan() {
    Mesh* pm = pmy_pack->pmesh;
    const int nmb_total = pm->nmb_total;
    const int nranks = global_variable::nranks;
    const int my_rank = global_variable::my_rank;
    const int first_gid = pmy_pack->gids;
    const int nmb_local = pmy_pack->nmb_thispack;
    const LogicalLocation* lloc = pm->lloc_eachmb;
    const int* rank_of = pm->rank_eachmb;
    const RegionIndcs& indcs = pm->mb_indcs;

    // Each MeshBlock's lower and upper edges along the scan direction (s) and
    // the two transverse directions (1, 2), in units of cells of the finest
    // level
    // ========================================================================
    // Find the finest resolved region in the entire mesh, not just this rank
    int finest = 0;
    for (int gid = 0; gid < nmb_total; ++gid) {
      finest = std::max(finest, lloc[gid].level);
    }

    std::vector<std::int64_t> lo_s(nmb_total), hi_s(nmb_total);
    std::vector<std::int64_t> lo1(nmb_total), hi1(nmb_total);
    std::vector<std::int64_t> lo2(nmb_total), hi2(nmb_total);
    // Compute the MeshBlock gid's edges along direction d
    auto SetEdges = [&](const int gid, const Direction d,
                        std::vector<std::int64_t>& lo,
                        std::vector<std::int64_t>& hi) {
      const LogicalLocation& loc = lloc[gid];
      const int cells = SelectWithDirection(d, indcs.nx1, indcs.nx2, indcs.nx3);
      // This is the size of the meshblock in units of the most refined cells,
      // cells * 2^(finest - loc.level)
      const std::int64_t size = std::int64_t{cells} << (finest - loc.level);

      // The starting location of the meshblock in units of the most refined
      // cells
      lo[gid] = size *
                SelectWithDirection<std::int64_t>(d, loc.lx1, loc.lx2, loc.lx3);

      // The ending location of the meshblock in units of the most refined cells
      hi[gid] = size + lo[gid];
    };
    for (int gid = 0; gid < nmb_total; ++gid) {
      SetEdges(gid, Dir, lo_s, hi_s);
      SetEdges(gid, t1_dir, lo1, hi1);
      SetEdges(gid, t2_dir, lo2, hi2);
    }
    // ========================================================================

    // Group the MeshBlocks by line of root (i.e. coarsest level) MeshBlocks,
    // which contains all of a MeshBlock's upstream MeshBlocks
    const int nroot1 = SelectWithDirection(t1_dir, pm->nmb_rootx1,
                                           pm->nmb_rootx2, pm->nmb_rootx3);
    const int nroot2 = SelectWithDirection(t2_dir, pm->nmb_rootx1,
                                           pm->nmb_rootx2, pm->nmb_rootx3);
    // n1 * 2^(finest - root_level) and n2 * 2^(finest - root_level)
    const std::int64_t root_size1 = std::int64_t{n1}
                                    << (finest - pm->root_level);
    const std::int64_t root_size2 = std::int64_t{n2}
                                    << (finest - pm->root_level);
    // Index of the line of root MeshBlocks containing MeshBlock gid
    auto RootLine = [&](const int gid) -> int {
      return static_cast<int>((lo2[gid] / root_size2) * nroot1 +
                              lo1[gid] / root_size1);
    };
    std::vector<std::vector<int>> root_lines(nroot1 * nroot2);
    for (int gid = 0; gid < nmb_total; ++gid) {
      root_lines[RootLine(gid)].push_back(gid);
    }

    // ========================================================================

    // The Source for upstream MeshBlock other of MeshBlock target, whose line
    // totals are at index face of faces or recvbuf
    auto MakeSource = [&](const int other, const int target,
                          const int face) -> Source {
      const int dlevel = lloc[other].level - lloc[target].level;
      const int fine = (dlevel >= 0) ? other : target;
      const int coarse = (dlevel >= 0) ? target : other;
      // Offsets of the finer face from the coarser one, in finer cells
      const int cell_shift = finest - lloc[fine].level;
      // The edge differences / 2^cell_shift
      const std::int64_t off1 = (lo1[fine] - lo1[coarse]) >> cell_shift;
      const std::int64_t off2 = (lo2[fine] - lo2[coarse]) >> cell_shift;
      if (off1 > INT32_MAX || off2 > INT32_MAX) {
        FatalError("LineScan: mesh refinement too deep for int offsets");
      }
      return {face, dlevel, static_cast<int>(off1), static_cast<int>(off2)};
    };

    // Whether MeshBlock other is upstream of MeshBlock target: their faces
    // overlap and other is entirely before target along the scan
    auto IsUpstream = [&](const int other, const int target) -> bool {
      const bool overlap1 =
          lo1[other] < hi1[target] && lo1[target] < hi1[other];
      const bool overlap2 =
          lo2[other] < hi2[target] && lo2[target] < hi2[other];
      const bool before = (Kind == ScanKind::Prefix)
                              ? hi_s[other] <= lo_s[target]
                              : lo_s[other] >= hi_s[target];
      return overlap1 && overlap2 && before;
    };
    // Each local MeshBlock's sources, in gid order, and the faces to send to
    // and receive from each rank. Remote sources hold their gid as face until
    // recvbuf is laid out
    std::vector<Source> local_srcs, remote_srcs;
    std::vector<std::set<int>> send_gids(nranks), recv_gids(nranks);
    h_local_start(0) = h_remote_start(0) = 0;
    for (int m = 0; m < nmb_local; ++m) {
      const int target = first_gid + m;
      for (const int other : root_lines[RootLine(target)]) {
        const int rank = rank_of[other];
        if (IsUpstream(other, target)) {
          if (rank == my_rank) {
            local_srcs.push_back(MakeSource(other, target, other - first_gid));
          } else {
            remote_srcs.push_back(MakeSource(other, target, other));
            recv_gids[rank].insert(other);
          }
        } else if (rank != my_rank && IsUpstream(target, other)) {
          send_gids[rank].insert(target);
        }
      }
      h_local_start(m + 1) = static_cast<int>(local_srcs.size());
      h_remote_start(m + 1) = static_cast<int>(remote_srcs.size());
    }

    // Lay out sendbuf and recvbuf by rank and then gid: fill peers with each
    // rank's faces and return the total number of faces
    auto LayOut = [&](const std::vector<std::set<int>>& gid_sets,
                      std::vector<Peer>& peers) -> int {
      peers.clear();
      int nfaces = 0;
      for (int rank = 0; rank < nranks; ++rank) {
        const int n = static_cast<int>(gid_sets[rank].size());
        if (n > 0) {
          peers.push_back({rank, nfaces, n});
          nfaces += n;
        }
      }
      return nfaces;
    };
    nsend_faces = LayOut(send_gids, send_peers);
    nrecv_faces = LayOut(recv_gids, recv_peers);

    // The local MeshBlock sent from each slot of sendbuf
    Reserve(h_send_m, nsend_faces, "LineScan::h_send_m");
    int send_slot = 0;
    for (const Peer& peer : send_peers) {
      for (const int gid : send_gids[peer.rank]) {
        h_send_m(send_slot++) = gid - first_gid;
      }
    }
    // Point the remote sources at their slots in recvbuf
    std::vector<int> recv_slot(nmb_total);
    for (const Peer& peer : recv_peers) {
      int slot = peer.first_face;
      for (const int gid : recv_gids[peer.rank]) recv_slot[gid] = slot++;
    }
    for (Source& src : remote_srcs) src.face = recv_slot[src.face];

    n_local_sources = static_cast<int>(local_srcs.size());
    n_remote_sources = static_cast<int>(remote_srcs.size());
    Reserve(h_local_sources, local_srcs.size(), "LineScan::h_local_sources");
    Reserve(h_remote_sources, remote_srcs.size(), "LineScan::h_remote_sources");
    std::copy(local_srcs.begin(), local_srcs.end(), h_local_sources.data());
    std::copy(remote_srcs.begin(), remote_srcs.end(), h_remote_sources.data());
  }

  /*!
   * \brief The new size for an array of the given size that must hold n
   * entries: at least double, so a slowly growing mesh rarely reallocates.
   */
  static std::size_t GeometricGrowth(const std::size_t size,
                                     const std::size_t n) {
    return Kokkos::max(n, 2 * size);
  }

  /*!
   * \brief Make a 1D array hold at least n entries. Reallocates, losing its
   * contents, only if it is too small.
   */
  template <typename T, typename Space>
  void Reserve(Kokkos::View<T*, LayoutWrapper, Space>& a, const std::size_t n,
               const std::string& label) {
    if (a.extent(0) >= n) return;
    a = Kokkos::View<T*, LayoutWrapper, Space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, label),
        GeometricGrowth(a.extent(0), n));
  }

  /*!
   * \brief Make sendbuf or recvbuf hold at least nfaces faces, as above.
   */
  void Reserve(DvceArray3D<Real>& a, const int nfaces,
               const std::string& label) {
    if (static_cast<int>(a.extent(0)) >= nfaces) return;
    a = DvceArray3D<Real>(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing, label),
        GeometricGrowth(a.extent(0), nfaces), n2, n1);
  }

  //--------------------------------------------------------------------------------------
  // Helpers

  /*!
   * \brief The x1, x2 or x3 value for direction I, J or K.
   */
  template <typename T>
  static T SelectWithDirection(const Direction d, const T& x1, const T& x2,
                               const T& x3) {
    if (d == Direction::I) return x1;
    if (d == Direction::J) return x2;
    return x3;
  }

  /*!
   * \brief Print msg and exit.
   */
  [[noreturn]] static void FatalError(const std::string& msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
              << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }
};
}  // namespace line_scan
#endif  // UTILS_SCAN_LINE_SCAN_HPP_
