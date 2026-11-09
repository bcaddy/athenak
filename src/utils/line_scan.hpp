//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file line_scan.hpp
//  \brief Header file for prefix and suffix sums along any cartesian
//  direction, inclusive, exclusive or anything in between as set by an
//  inclusive fraction

#ifndef UTILS_LINE_SCAN_HPP_
#define UTILS_LINE_SCAN_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "parameter_input.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif  // MPI_PARALLEL_ENABLED

namespace line_scan {
/// The direction of the scan
enum class Direction { I, J, K };

/// The type of scan to perform, prefix or suffix
enum class ScanKind { Prefix, Suffix };

/// The stage a LineScan currently has in flight
enum class Stage {
  NotStarted,
  BlockLocalScan,
  Exchange,
  AddUpstream,
  Completed
};

//----------------------------------------------------------------------------------------
/*!
 * \brief Drive several LineScans round-robin until all are completed.
 *
 * \details Fence all prior device work so the scans' inputs are ready, reset
 * every scan, then loop over them to progress each scan so the scans advance
 * independently and overlap. Every rank must pass the scans in the same order.
 *
 * \tparam Scans The types of the scans, any mix of LineScans (or anything with
 * the same Reset and Driver methods).
 * \param scans The scans to drive.
 */
template <typename... Scans>
void MultiScanDriver(Scans&&... scans) {
  // The scans run on their own execution space instances, so their inputs
  // must be ready before they start
  Kokkos::fence();

  // Reset all the scans
  (scans.Reset(), ...);

  // flag to check if scans are complete
  bool all_completed = false;

  // Loop over all scans until they're complete
  while (!all_completed) {
    all_completed = true;

    // Call the driver function on each scan. If any scan isn't complete then
    // run the outer loop again. Not &&, which would skip the remaining scans
    // once one isn't complete
    ((all_completed &= (scans.Driver() == Stage::Completed)), ...);
  }
}

//----------------------------------------------------------------------------------------
/*!
 * \class LineScan
 * \brief Compute either prefix sums or suffix sums of one or more fields
 * along a given axis across the entire domain, inclusive, exclusive or
 * anything in between as set by inclusive_fraction
 *
 * \details Each use writes the global scan of every field to scan_data: each
 * MeshBlock's block local scan plus its upstream sum, the running total of the
 * MeshBlocks upstream of it on any rank. The domain boundaries, periodic or
 * not, start the upstream sum at zero.
 *
 * Each cell's entry is the exclusive scan, the sum of the cells upstream of
 * it, plus inclusive_fraction times the cell's own value: 0 gives an exclusive
 * scan, 1 an inclusive one, and the default 0.5; i.e. the sum up to the cell's
 * center. Only the cell's own entry is affected; the line totals passed
 * downstream always include the whole cell. inclusive_fraction is read from
 * shielding_fraction_cell in the input block of the module that owns the
 * scan.
 *
 * The running total is passed along each line from MeshBlock to MeshBlock, as
 * a chain: a MeshBlock B's outgoing total, its upstream sum plus its own line
 * totals, is added to the upstream sum of each of its immediate downstream
 * face neighbours, mapped to that neighbour's resolution. So B's upstream sum
 * is the sum of the outgoing totals of its immediate upstream face
 * neighbours. On a uniform mesh that is the sum of the line totals of every
 * MeshBlock upstream.
 *
 * With mesh refinement, face neighbours differ by at most one level, so every
 * step of the chain maps by a factor of 2: a finer neighbour's outgoing total
 * is area averaged (2x2 cells per coarse cell) and a coarser one linearly
 * prolongated with minmod slopes (one sided at face edges, scaled to stay
 * non-negative). Coarser upstream totals are therefore summed first and
 * prolongated once, at each coarse-to-fine face, rather than each upstream
 * MeshBlock being prolongated separately. The two agree exactly (up to
 * rounding) on uniform meshes and wherever no coarse-to-fine face is upstream,
 * since restriction composes, and they conserve the same total in every coarse
 * cell, but the slopes and limiter now see the aggregate. Structure from finer
 * MeshBlocks that passes through coarser ones reaches a finer MeshBlock
 * downstream only at the coarser resolution. The upstream sums stay
 * non-negative.
 *
 * The Num_Fields fields of a LineScan share its exchange plan, execution space
 * instance, MPI communicator and messages (each carries every field), and
 * value_func returns all of a cell's fields from one call, so scanning fields
 * together is cheaper than in separate LineScans. Prefix and suffix scans need
 * separate LineScans. scan_data is indexed (m, field_idx, k, j, i);
 * Kokkos::subview(scan_data, Kokkos::ALL, field_idx, Kokkos::ALL, Kokkos::ALL,
 * Kokkos::ALL) is one field (LayoutStride, not a DvceArray4D).
 *
 * The exchange plan is built from each MeshBlock's own neighbour data, so its
 * cost and the message sizes grow with the number of MeshBlocks on the rank
 * and its surface, not with the whole mesh. Messages to one rank are batched
 * per face plane, and a rank sends a batch as soon as the MeshBlocks in it are
 * complete, so lines pipeline across ranks.
 *
 * Each LineScan runs its kernels on its own execution space instance and has
 * its own MPI communicator, so several can run concurrently. `scan_data` is
 * ready once Driver returns Completed. The kernel methods are public only
 * because nvcc requires that of methods containing a `KOKKOS_LAMBDA`; their
 * Impl prefix marks them as internal.
 *
 * Requirements on the caller:
 *  - All GPU operations that `value_func` depends on must be complete before
 * starting a scan. Easiest done by calling a Kokkos Fence before running scans
 *  - Every rank constructs LineScans in the same order, since the constructor
 *    duplicates a communicator, which is collective.
 *  - Started scans are driven to completion on every rank, polled round-robin
 *    or completed in the same order everywhere, or ranks can deadlock. A rank
 *    passes its upstream totals on only while its scan is being driven, so
 *    keep driving a scan until it is Completed.
 *  - No scan is in flight when the mesh is refined.
 *  - LineScans are destroyed before MPI and Kokkos are finalized.
 *  - LineScans are destroyed in the same order on every rank, since the
 *    destructor frees a communicator, which is collective.
 *  - value_func only captures data that persists across mesh refinement, e.g.
 *    u0 or w0; MeshBlock geometry is passed to it instead. With mesh
 *    refinement every field's value must be extensive (e.g. density times dx)
 *    and non-negative.
 *
 * Since Dir, Kind and Num_Fields are given explicitly, ValueFunc must be too,
 * e.g. with decltype:
 * \code
 * auto func = KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
 *                           const RegionSize& size) -> Kokkos::Array<Real, 2> {
 *   const Real dens = w0(m, IDN, k, j, i);
 *   return {dens * size.dx1, dens * dens * size.dx1};
 * };
 * line_scan::LineScan<line_scan::Direction::I, line_scan::ScanKind::Prefix, 2,
 *                     decltype(func)> scan(pmbp, func, pin, "chemistry");
 * \endcode
 *
 * \tparam Dir The direction along which the scan is performed.
 * \tparam Kind Whether to compute a prefix or suffix sum.
 * \tparam Num_Fields The number of fields scanned, at least 1.
 * \tparam ValueFunc Type of a device callable returning the values of all the
 * fields to sum at a cell, with the signature
 * \code
 * KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
 *               const RegionSize& size) -> Kokkos::Array<Real, Num_Fields>
 * \endcode
 * where m is the MeshBlock's index in the MeshBlockPack, k, j, i are its cell
 * indices (mb_indcs, including ghost cells), and size is its mb_size entry.
 * It returns all the fields from one call, and is called once per cell.
 *
 * \note Only sums are supported. Another associative operation would need its
 * identity in place of 0, a Kokkos reducer in ImplBlockLocalScan_I, and its own
 * restriction and prolongation in the exchange.
 */
template <Direction Dir, ScanKind Kind, int Num_Fields, typename ValueFunc>
class LineScan {
  static_assert(Num_Fields >= 1, "LineScan needs at least one field");

 public:
  /*!
   * \brief Allocate the scan's arrays and duplicate MPI_COMM_WORLD, which is
   * collective over all ranks.
   * \param ppack The MeshBlockPack to scan over.
   * \param value_func Returns the values of all the fields to sum at each
   * cell.
   * \param pin The ParameterInput object
   * \param block The input block of the physics module that owns this scan,
   * from which shielding_fraction_cell, the fraction of each cell's own
   * value included in its entry of scan_data (0 for an exclusive scan, 1 for an
   * inclusive one), is read. Defaults to 0.5.
   */
  LineScan(MeshBlockPack* ppack, const ValueFunc& value_func,
           ParameterInput* pin, const std::string& block)
      // Not initialized, since every scan writes all the entries it uses
      : scan_data(Kokkos::view_alloc(Kokkos::WithoutInitializing, "scan_data"),
                  std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank),
                  Num_Fields, ppack->pmesh->mb_indcs.nx3,
                  ppack->pmesh->mb_indcs.nx2, ppack->pmesh->mb_indcs.nx1),
        pmy_pack(ppack),
        value_func(value_func),
        inclusive_fraction(
            pin->GetOrAddReal(block, "shielding_fraction_cell", 0.5)),
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
        r1(RestrictFactor(n1)),
        r2(RestrictFactor(n2)),
        h1(n1 / r1),
        h2(n2 / r2),
        faces(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "LineScan::faces"),
            nmb, Num_Fields, n2, n1),
        upstream_sum(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        "LineScan::upstream_sum"),
                     nmb, Num_Fields, n2, n1),
        // The plan and message arrays, at their bounds (see Plan bounds). A
        // single rank sends no messages
        items(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "LineScan::items"),
            MaxItems()),
        sources(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "LineScan::sources"),
                MaxSources()),
        ports(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "LineScan::ports"),
            MaxPorts()),
        sendbuf(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "LineScan::sendbuf"),
                (global_variable::nranks > 1) ? MaxPorts() * FaceReals() : 0),
        recvbuf(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "LineScan::recvbuf"),
                (global_variable::nranks > 1) ? nmb * FaceReals() : 0),
        h_items(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "LineScan::h_items"),
                MaxItems()),
        h_sources(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                     "LineScan::h_sources"),
                  MaxSources()),
        h_ports(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "LineScan::h_ports"),
                MaxPorts()) {
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
   * \brief Advance the scan, without blocking.
   * \return The current stage. Once Completed, scan_data can be read without
   * a fence.
   */
  Stage Driver() {
    switch (stage) {
      // Run the block local scans, build the exchange plan and sum each
      // MeshBlock's local upstream line totals
      case Stage::NotStarted:
        // Choose the correct block local scan for the direction. Launched
        // first since it doesn't use the exchange plan, so the device runs it
        // while the host builds the plan
        if constexpr (Dir == Direction::I) {
          ImplBlockLocalScan_I();
        } else {
          ImplBlockLocalScan_JK();
        }

        // Rebuild the plan on the first run and after the mesh changes
        if (plan_mesh_seq != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
          BuildExchangePlan();
        }

        // If MPI is enabled then post the receives now
        PostReceives();

        // Some MeshBlocks send data to other ranks that doesn't depend on any
        // message this rank receives. Sum and pack those MeshBlocks first, so
        // their sends can be posted as soon as possible. The rest of the local
        // pass runs on the same stream, so it is launched only after those
        // sends are posted; otherwise the check that the packing has finished
        // would also wait for it
        ImplAccumulate(0, (n_first_items > 0) ? n_first_items : n_local_items,
                       /*overwrite=*/true);

        // Update the stage and break
        stage = Stage::BlockLocalScan;
        break;
      case Stage::BlockLocalScan:
        if (n_first_items > 0) {
          // Check if the send buffer is packed yet. If not then immediately
          // return
          if (!ExecSpaceIdle(exec_space)) break;
          PostSends(first_sends);

          // Run the rest of the local pass while the messages are in flight
          ImplAccumulate(n_first_items, n_local_items - n_first_items,
                         /*overwrite=*/true);
        }
        StartExchange();
        stage = Stage::Exchange;
        break;
      case Stage::Exchange:
        // Run each exchange step once its inputs have arrived, and return
        // while any is still waiting
        if (!ProgressExchange()) break;
        // The MeshBlocks that nothing else reads are summed last, together
        ImplAccumulate(deferred_begin, deferred_end - deferred_begin,
                       /*overwrite=*/false);
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

  // The global scan of each field, inclusive, exclusive or anything in
  // between as set by inclusive_fraction, indexed (m, field_idx, k, j, i) over
  // each MeshBlock's real cells from 0
  const DvceArray5D<Real> scan_data;

  //--------------------------------------------------------------------------------------
  // Kernels. Public only because nvcc requires that of methods containing a
  // KOKKOS_LAMBDA; the Impl prefix marks them as not for use outside LineScan
  // and they should be treated as private.

  // An upstream face added to a MeshBlock's upstream sum, mapped to its
  // resolution: a local MeshBlock's line totals (faces) or outgoing total
  // (upstream_sum plus faces), or a face received in recvbuf. Faces mapped
  // between levels are immediate neighbours, so one level apart. Public only
  // because nvcc forbids capturing private or protected types in a
  // KOKKOS_LAMBDA
  struct Source {
    int m = -1;              // local MeshBlock read, or -1 to read recvbuf
    bool out = false;        // whether to read m's outgoing total, not faces
    std::size_t offset = 0;  // start in recvbuf of the received face
    bool quarter = false;    // whether the received face is a quarter face
    int dlevel = 0;          // source level minus target level: -1, 0 or 1
    // Offset of the finer face's quadrant in the coarser face, in coarser
    // cells. For dlevel 0, of the source face in the target face
    int off1 = 0, off2 = 0;
  };

  // A slot of sendbuf holding a MeshBlock's outgoing total for a receiving rank
  struct Port {
    std::size_t offset;  // start in sendbuf
    bool coarser;  // whether the receiver is coarser, so gets it restricted
  };

  // A MeshBlock that one ImplAccumulate launch sums into upstream_sum, packs
  // into its Ports, or both
  struct Item {
    int m;                     // local MeshBlock
    int src_begin, src_end;    // its Sources in sources
    int port_begin, port_end;  // its Ports in ports
  };

  // One field of a source face of n1 x n2 cells: data(c2 * n1 + c1), plus the
  // same entry of add if it isn't null, to read the outgoing total faces +
  // upstream_sum
  struct FaceRef {
    const Real* data;
    const Real* add;
    int n1, n2;

    KOKKOS_INLINE_FUNCTION
    Real operator()(const int c2, const int c1) const {
      const std::size_t c = static_cast<std::size_t>(c2) * n1 + c1;
      return (add == nullptr) ? data[c] : data[c] + add[c];
    }
  };

  /*!
   * \brief Scan along i of each field of value_func over each MeshBlock's
   * real cells, inclusive, exclusive or anything in between as set by
   * inclusive_fraction, with the line totals stored in faces.
   *
   * Each (m, k, j) line is one team thread, whose vector lanes split the line.
   * This suits short lines (nx1 <= 32), where a whole team per line would
   * leave most of its threads idle. value_func runs once per cell, into the
   * thread's scratch memory, and each field is then scanned from there.
   */
  void ImplBlockLocalScan_I() {
    // Local copies so the device lambda doesn't capture `this`
    auto scan_data_ = scan_data;
    auto faces_ = faces;
    auto value_func_ = value_func;
    const Real frac = inclusive_fraction;
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

    // Each thread's values of every field along its line, in fast scratch
    // memory if it fits and otherwise in device memory
    const std::size_t scr_bytes = ScrArray2D<Real>::shmem_size(Num_Fields, nx);
    const std::size_t scr_max = Kokkos::TeamPolicy<>::scratch_size_max(0);
    const int scr_level = (scr_bytes * team_size <= scr_max) ? 0 : 1;

    Kokkos::parallel_for(
        "BlockLocalScan_I",
        Kokkos::TeamPolicy<>(exec_space, n_league, team_size, vlen)
            .set_scratch_size(scr_level, Kokkos::PerThread(scr_bytes)),
        KOKKOS_LAMBDA(TeamMember_t t) {
          // Every thread reaches the barrier below, so the extra threads of
          // the last team skip the work instead of returning
          const int line = t.league_rank() * t.team_size() + t.team_rank();
          const bool active = line < nlines;
          const int m = line / (nk * nj);
          const int k = (line / nj) % nk;
          const int j = line % nj;
          ScrArray2D<Real> values(t.thread_scratch(scr_level), Num_Fields, nx);

          // The values of all the fields at each cell
          if (active) {
            const RegionSize size = mb_size(m);
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(t, nx), [=](const int i) {
                  const auto x = value_func_(m, k + ks, j + js, i + is, size);
                  for (int field_idx = 0; field_idx < Num_Fields; ++field_idx) {
                    values(field_idx, i) = x[field_idx];
                  }
                });
          }
          // The vector lanes read each other's values below
          t.team_barrier();
          if (!active) return;

          // The range starts at 0 since the CUDA vector scan ignores a nonzero
          // start
          for (int field_idx = 0; field_idx < Num_Fields; ++field_idx) {
            Real total;
            Kokkos::parallel_scan(
                Kokkos::ThreadVectorRange(t, nx),
                [=](const int p, Real& update, const bool final) {
                  const int i = (Kind == ScanKind::Suffix) ? nx - 1 - p : p;
                  const Real x = values(field_idx, i);
                  if (final) {
                    scan_data_(m, field_idx, k, j, i) = update + frac * x;
                  }
                  update += x;
                },
                total);
            Kokkos::single(Kokkos::PerThread(t),
                           [=]() { faces_(m, field_idx, k, j) = total; });
          }
        });
  }

  /*!
   * \brief Scan along j or k, inclusive, exclusive or anything in between as
   * set by inclusive_fraction, stored as in ImplBlockLocalScan_I.
   *
   * Each line is one thread stepping along it, keeping every field's sum, so
   * value_func runs once per cell and neighbouring threads (neighbouring i)
   * access memory coalesced. This is 3-9x slower than ImplBlockLocalScan_I
   * for scans along i.
   */
  void ImplBlockLocalScan_JK() {
    auto scan_data_ = scan_data;
    auto faces_ = faces;
    auto value_func_ = value_func;
    const Real frac = inclusive_fraction;
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
          Real sum[Num_Fields] = {};
          for (int s = 0; s < n; ++s) {
            const int p = (Kind == ScanKind::Suffix) ? n - 1 - s : s;
            const int kp = (Dir == Direction::K) ? p : k;
            const int jp = (Dir == Direction::J) ? p : j;
            const auto x = value_func_(m, kp + ks, jp + js, i + is, size);
            for (int field_idx = 0; field_idx < Num_Fields; ++field_idx) {
              scan_data_(m, field_idx, kp, jp, i) =
                  sum[field_idx] + frac * x[field_idx];
              sum[field_idx] += x[field_idx];
            }
          }
          // The face cell is (k, i) along j and (j, i) along k
          for (int field_idx = 0; field_idx < Num_Fields; ++field_idx) {
            faces_(m, field_idx, (Dir == Direction::J) ? k : j, i) =
                sum[field_idx];
          }
        });
  }

  /*!
   * \brief Sum the Sources of items [first, first + count) into their
   * MeshBlocks' upstream_sum, then copy each one's outgoing total,
   * upstream_sum plus faces, into its Ports, so a MeshBlock's sends are packed
   * by the kernel that completes it. A Port to a coarser receiver gets the
   * total restricted 2x2
   * (2 cells in 2D, 1 in 1D), as flux correction does, a quarter of the face.
   * Each thread does the r1 x r2 face cells of one restricted cell, for every
   * item.
   * \param first, count The items.
   * \param overwrite Whether the sums replace upstream_sum, in the local pass,
   * or add to it.
   */
  void ImplAccumulate(const int first, const int count, const bool overwrite) {
    if (count == 0) return;
    auto items_ = items;
    auto sources_ = sources;
    auto ports_ = ports;
    const Real* const faces_ptr = faces.data();
    Real* const upstream_ptr = upstream_sum.data();
    const Real* const recv_ptr = recvbuf.data();
    Real* const send_ptr = sendbuf.data();
    const int n1_ = n1, n2_ = n2, r1_ = r1, r2_ = r2, h1_ = h1, h2_ = h2;
    par_for(
        "LineScan::Accumulate", exec_space, 0, count - 1, 0, Num_Fields - 1, 0,
        h2_ - 1, 0, h1_ - 1,
        KOKKOS_LAMBDA(const int p, const int field_idx, const int j2,
                      const int j1) {
          const Item item = items_(first + p);
          // This field of the MeshBlock's face in faces and upstream_sum
          const std::size_t face =
              (static_cast<std::size_t>(item.m) * Num_Fields + field_idx) *
              n2_ * n1_;
          Real out[2][2];
          for (int a2 = 0; a2 < r2_; ++a2) {
            for (int a1 = 0; a1 < r1_; ++a1) {
              const std::size_t cell =
                  face + static_cast<std::size_t>(j2 * r2_ + a2) * n1_ +
                  j1 * r1_ + a1;
              out[a2][a1] = overwrite ? 0.0 : upstream_ptr[cell];
            }
          }
          for (int s = item.src_begin; s < item.src_end; ++s) {
            const Source src = sources_(s);
            FaceRef ref;
            if (src.m >= 0) {
              const std::size_t src_face =
                  (static_cast<std::size_t>(src.m) * Num_Fields + field_idx) *
                  n2_ * n1_;
              ref = {faces_ptr + src_face,
                     src.out ? upstream_ptr + src_face : nullptr, n1_, n2_};
            } else {
              const int m1 = src.quarter ? h1_ : n1_;
              const int m2 = src.quarter ? h2_ : n2_;
              ref = {recv_ptr + src.offset +
                         static_cast<std::size_t>(field_idx) * m2 * m1,
                     nullptr, m1, m2};
            }
            for (int a2 = 0; a2 < r2_; ++a2) {
              for (int a1 = 0; a1 < r1_; ++a1) {
                out[a2][a1] +=
                    MapToCell(src, ref, j1 * r1_ + a1, j2 * r2_ + a2, r1_, r2_);
              }
            }
          }
          for (int a2 = 0; a2 < r2_; ++a2) {
            for (int a1 = 0; a1 < r1_; ++a1) {
              const std::size_t cell =
                  face + static_cast<std::size_t>(j2 * r2_ + a2) * n1_ +
                  j1 * r1_ + a1;
              // A pack-only item leaves upstream_sum to the kernel that
              // completed it
              if (overwrite || item.src_end > item.src_begin) {
                upstream_ptr[cell] = out[a2][a1];
              }
              out[a2][a1] += faces_ptr[cell];
            }
          }
          for (int q = item.port_begin; q < item.port_end; ++q) {
            const Port port = ports_(q);
            if (port.coarser) {
              Real sum = 0.0;
              for (int a2 = 0; a2 < r2_; ++a2) {
                for (int a1 = 0; a1 < r1_; ++a1) {
                  sum += out[a2][a1];
                }
              }
              send_ptr[port.offset +
                       (static_cast<std::size_t>(field_idx) * h2_ + j2) * h1_ +
                       j1] = sum / static_cast<Real>(r1_ * r2_);
            } else {
              for (int a2 = 0; a2 < r2_; ++a2) {
                for (int a1 = 0; a1 < r1_; ++a1) {
                  const int t1 = j1 * r1_ + a1, t2 = j2 * r2_ + a2;
                  send_ptr[port.offset +
                           (static_cast<std::size_t>(field_idx) * n2_ + t2) *
                               n1_ +
                           t1] = out[a2][a1];
                }
              }
            }
          }
        });
  }

  /*!
   * \brief Add each MeshBlock's upstream sum to its block local scan.
   */
  void ImplAddUpstream() {
    if (pmy_pack->pmesh->mb_indcs.nx1 % 2 == 0) {
      ImplAddUpstreamKernel<2>();
    } else {
      ImplAddUpstreamKernel<1>();
    }
  }

  /*!
   * \brief ImplAddUpstream with N neighbouring cells per thread as one access.
   * Assumes each rank holds fewer than 2^31 scan_data entries (cells times
   * fields; cbroot(2^31) ~= 1290^3). N =
   * 1 is used for meshblocks with an odd number of cells and N = 2 is used for
   * meshblocks with an even number. N = 2 is about 1.4x faster than N = 1
   */
  template <int N>
  void ImplAddUpstreamKernel() {
    // N adjacent cells, loaded and stored as one aligned access
    struct alignas(N * sizeof(Real)) RealPack {
      Real data[N];
    };

    // Collect class data
    auto& indcs = pmy_pack->pmesh->mb_indcs;
    const std::uint32_t nx1 = indcs.nx1;
    const std::uint32_t n21 = nx1 * indcs.nx2;
    const std::uint32_t n321 = n21 * indcs.nx3;
    Real* const scan_data_ptr = scan_data.data();
    const Real* const upstream_sum_ptr = upstream_sum.data();

    // Perform the add
    par_for(
        "LineScan::AddUpstream", exec_space, 0,
        static_cast<int>(n321 * pmy_pack->nmb_thispack * Num_Fields / N) - 1,
        KOKKOS_LAMBDA(const int p) {
          // The offset into the scan_data array
          const std::uint32_t scan_data_offset =
              static_cast<std::uint32_t>(p) * N;

          // The scan data to update
          RealPack scan_loaded = *reinterpret_cast<const RealPack*>(
              scan_data_ptr + scan_data_offset);

          // Now perform the add based on the direction
          if (Dir == Direction::I) {
            const Real upstream_loaded =
                upstream_sum_ptr[scan_data_offset / nx1];
            for (int i = 0; i < N; ++i) {
              scan_loaded.data[i] += upstream_loaded;
            }
          } else if (Dir == Direction::J) {
            const std::uint32_t upstream_offset =
                scan_data_offset / n21 * nx1 + scan_data_offset % nx1;
            const RealPack upstream_loaded = *reinterpret_cast<const RealPack*>(
                upstream_sum_ptr + upstream_offset);
            for (int i = 0; i < N; ++i) {
              scan_loaded.data[i] += upstream_loaded.data[i];
            }
          } else {
            const std::uint32_t upstream_offset =
                scan_data_offset / n321 * n21 + scan_data_offset % n21;
            const RealPack upstream_loaded = *reinterpret_cast<const RealPack*>(
                upstream_sum_ptr + upstream_offset);
            for (int i = 0; i < N; ++i) {
              scan_loaded.data[i] += upstream_loaded.data[i];
            }
          }

          // Write out the updates to scan_data
          *reinterpret_cast<RealPack*>(scan_data_ptr + scan_data_offset) =
              scan_loaded;
        });
  }

 private:
  // A face neighbour of a local MeshBlock, from nghbr
  struct Neighbor {
    int gid, rank, level;
    int sub;  // its sub-slot in nghbr, (t1 quadrant) + 2 (t2 quadrant) if finer
  };

  // A message of the exchange plan, between this rank and rank: the faces of
  // the MeshBlocks on one face plane, all fields of each, in slots ordered by
  // the gid of the MeshBlock that sends them
  struct Batch {
    int rank;               // the other rank
    int tag;                // index of the plane among those of this pair
    std::size_t offset;     // start in sendbuf or recvbuf, in Reals
    std::size_t count;      // Reals
    std::vector<int> gids;  // the sending MeshBlock of each slot
  };

  // A group of exchange work that runs once its inputs are ready: the
  // receives it reads have arrived and the steps it reads from have been
  // launched (their kernels then run first, on the same stream). It launches
  // one ImplAccumulate of the MeshBlocks that other ranks or steps read, which
  // packs its sends, and posts them once the stream is idle
  struct Step {
    std::vector<int> recvs;    // receives it waits for
    std::vector<int> after;    // steps launched before it
    int item_begin, item_end;  // its items
    std::vector<int> sends;    // sends it packs
  };

  // The two transverse directions, across the scan, in increasing order
  static constexpr Direction t1_dir =
      (Dir == Direction::I) ? Direction::J : Direction::I;
  static constexpr Direction t2_dir =
      (Dir == Direction::K) ? Direction::J : Direction::K;
  // The scan direction as 0, 1, 2 for x1, x2, x3
  static constexpr int s_axis =
      (Dir == Direction::I) ? 0 : ((Dir == Direction::J) ? 1 : 2);

  MeshBlockPack* pmy_pack;
  const ValueFunc value_func;
  // The fraction of each cell's own value in its entry of scan_data
  const Real inclusive_fraction;
  // Declared before the Views it allocates
  const DevExeSpace exec_space;

  // MeshBlocks in scan_data
  const int nmb;
  // Real cells along the scan and the two transverse directions (1 if unused)
  const int n_scan, n1, n2;
  // Restriction factor along each transverse direction (see RestrictFactor),
  // and the cells across a restricted face, n / r
  const int r1, r2, h1, h2;

  // Each MeshBlock's line totals, and its upstream sum, of every field.
  // Indexed (m, field_idx, t2, t1)
  const DvceArray4D<Real> faces, upstream_sum;

  Stage stage = Stage::NotStarted;

  //--------------------------------------------------------------------------------------
  // Exchange plan, built by Driver when the mesh changes

  // The ImplAccumulate launches' items, Sources and Ports. Items
  // [0, n_local_items) are the local pass, every MeshBlock once, of which
  // [0, n_first_items) pack first_sends; then each Step's items; then
  // [deferred_begin, deferred_end), summed after every step
  const DvceArray1D<Item> items;
  const DvceArray1D<Source> sources;
  const DvceArray1D<Port> ports;
  // Messages, laid out batch by batch in Reals
  const DvceArray1D<Real> sendbuf, recvbuf;
  // Host copies of the plan, kept alive for the asynchronous copies. Only a
  // prefix of each is in use
  const HostArray1D<Item> h_items;
  const HostArray1D<Source> h_sources;
  const HostArray1D<Port> h_ports;
  int n_local_items = 0, n_first_items = 0;
  int deferred_begin = 0, deferred_end = 0;

  std::vector<Batch> send_batches, recv_batches;
  std::vector<int> first_sends;  // sends packed by the local pass
  std::vector<Step> steps;

  // Progress through the steps in the current run
  std::vector<char> step_launched, recv_done;
  int packing_step = -1;  // a step whose sends await the packing kernel

  int plan_mesh_seq = -1;  // mesh update the plan was built for

#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  std::vector<MPI_Request> send_reqs, recv_reqs;
#endif  // MPI_PARALLEL_ENABLED

  //--------------------------------------------------------------------------------------
  // Mapping faces between neighbouring refinement levels

  /*!
   * \brief A source face mapped to target face cell (t1, t2), 0 outside the
   * part of the target face the source covers. Faces map between levels only
   * from an immediate neighbour, one level apart, so every map is by a factor
   * of 2 (1 along an unused direction). Coarser totals reach a finer
   * MeshBlock already summed, and are prolongated together: unlike prolonging
   * each upstream MeshBlock separately, the slopes and limiter see the sum, and
   * structure from finer MeshBlocks that passed through a coarser one arrives
   * only at the coarser resolution (see the class docs).
   * \param src The source.
   * \param face This field of the source face.
   * \param t1, t2 The target face cell.
   * \param r1, r2 The restriction factors along t1 and t2.
   */
  KOKKOS_INLINE_FUNCTION
  static Real MapToCell(const Source& src, const FaceRef& face, const int t1,
                        const int t2, const int r1, const int r2) {
    if (src.dlevel == 0) {
      const int c1 = t1 - src.off1, c2 = t2 - src.off2;
      if (c1 < 0 || c1 >= face.n1 || c2 < 0 || c2 >= face.n2) return 0.0;
      return face(c2, c1);
    }
    if (src.dlevel > 0) {
      return RestrictToCell(src, face, t1, t2, r1, r2);
    }
    return ProlongToCell(src, face, t1, t2, r1, r2);
  }

  /*!
   * \brief Area average of the 2x2 finer source cells inside target cell
   * (t1, t2) (2 cells in 2D, 1 in 1D), if it is in the source's quadrant.
   * Arguments as in MapToCell.
   */
  KOKKOS_INLINE_FUNCTION
  static Real RestrictToCell(const Source& src, const FaceRef& face,
                             const int t1, const int t2, const int r1,
                             const int r2) {
    const int d1 = t1 - src.off1, d2 = t2 - src.off2;
    if (d1 < 0 || d1 * r1 >= face.n1 || d2 < 0 || d2 * r2 >= face.n2) {
      return 0.0;
    }
    Real sum = 0.0;
    for (int c2 = d2 * r2; c2 < (d2 + 1) * r2; ++c2) {
      for (int c1 = d1 * r1; c1 < (d1 + 1) * r1; ++c1) {
        sum += face(c2, c1);
      }
    }
    return sum / static_cast<Real>(r1 * r2);
  }

  /*!
   * \brief Linear prolongation of a coarser source face, the sum of all the
   * totals upstream of it, to the center of target cell (t1, t2), a quarter
   * of a source cell from its center along each used direction. The minmod
   * slopes are one sided at the source face's edges, so each line stays
   * independent of the lines beside it. Only there can the profile go
   * negative (a centred minmod slope is at most the cell's value), so only
   * there are the slopes scaled to keep it non-negative. Prolongation then
   * restriction gives back the source cell, so totals are conserved per
   * coarse cell. Since the source is already the sum over every upstream
   * MeshBlock, rather than each being prolongated with its own slopes, finer
   * structure that passed through a coarser MeshBlock is not recovered here.
   * Arguments as in MapToCell.
   */
  KOKKOS_INLINE_FUNCTION
  static Real ProlongToCell(const Source& src, const FaceRef& face,
                            const int t1, const int t2, const int r1,
                            const int r2) {
    // The source cell containing the target cell, and the target cell
    // center's position in it, -1/4 or 1/4 of a source cell
    const int c1 = src.off1 + t1 / r1, c2 = src.off2 + t2 / r2;
    const Real xi1 = (r1 == 1) ? 0.0 : ((t1 % 2 == 1) ? 0.25 : -0.25);
    const Real xi2 = (r2 == 1) ? 0.0 : ((t2 % 2 == 1) ? 0.25 : -0.25);

    const Real u = face(c2, c1);
    Real s1 = 0.0, s2 = 0.0;
    bool edge = false;
    if (r1 > 1) {
      const bool has_left = c1 > 0, has_right = c1 < face.n1 - 1;
      s1 = Slope(has_left ? u - face(c2, c1 - 1) : 0.0,
                 has_right ? face(c2, c1 + 1) - u : 0.0, has_left, has_right);
      edge = edge || !(has_left && has_right);
    }
    if (r2 > 1) {
      const bool has_left = c2 > 0, has_right = c2 < face.n2 - 1;
      s2 = Slope(has_left ? u - face(c2 - 1, c1) : 0.0,
                 has_right ? face(c2 + 1, c1) - u : 0.0, has_left, has_right);
      edge = edge || !(has_left && has_right);
    }
    // Scale the slopes so the profile's minimum, u - (|s1| + |s2|) / 2, isn't
    // negative
    if (edge) {
      const Real slope_sum = fabs(s1) + fabs(s2);
      if (slope_sum > 2.0 * u) {
        const Real scale = fmax(2.0 * u, 0.0) / slope_sum;
        s1 *= scale;
        s2 *= scale;
      }
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
  // Running the exchange, without blocking

  /*!
   * \brief Post a receive into recvbuf for each receive batch.
   */
  void PostReceives() {
#if MPI_PARALLEL_ENABLED
    for (std::size_t b = 0; b < recv_batches.size(); ++b) {
      const Batch& batch = recv_batches[b];
      MPI_Irecv(recvbuf.data() + batch.offset, static_cast<int>(batch.count),
                MPI_ATHENA_REAL, batch.rank, batch.tag, comm, &recv_reqs[b]);
    }
#endif  // MPI_PARALLEL_ENABLED
  }

  /*!
   * \brief Send the given send batches, already packed in sendbuf.
   */
  void PostSends(const std::vector<int>& batch_ids) {
#if MPI_PARALLEL_ENABLED
    for (const int b : batch_ids) {
      const Batch& batch = send_batches[b];
      MPI_Isend(sendbuf.data() + batch.offset, static_cast<int>(batch.count),
                MPI_ATHENA_REAL, batch.rank, batch.tag, comm, &send_reqs[b]);
    }
#else
    (void)batch_ids;
#endif  // MPI_PARALLEL_ENABLED
  }

  /*!
   * \brief Mark every step unlaunched and every receive pending.
   */
  void StartExchange() {
    step_launched.assign(steps.size(), 0);
    recv_done.assign(recv_batches.size(), 0);
    packing_step = -1;
  }

  /*!
   * \brief Launch every step whose inputs are ready, in plan order but as
   * dataflow, so a step never waits for an unrelated one. A step that packs
   * sends is followed only by posting them, once the stream is idle, so the
   * sends aren't delayed by kernels queued after the packing one.
   * \return Whether every step has been launched and its sends posted.
   */
  bool ProgressExchange() {
    while (true) {
      if (packing_step >= 0) {
        if (!ExecSpaceIdle(exec_space)) return false;
        PostSends(steps[packing_step].sends);
        packing_step = -1;
      }
      TestReceives();

      // The first unlaunched step whose receives have arrived and whose
      // predecessors have been launched
      int ready = -1;
      bool all_launched = true;
      for (int s = 0; s < static_cast<int>(steps.size()) && ready < 0; ++s) {
        if (step_launched[s]) continue;
        all_launched = false;
        bool inputs = true;
        for (const int r : steps[s].recvs) inputs = inputs && recv_done[r];
        for (const int a : steps[s].after) inputs = inputs && step_launched[a];
        if (inputs) ready = s;
      }
      if (ready < 0) return all_launched;

      const Step& step = steps[ready];
      ImplAccumulate(step.item_begin, step.item_end - step.item_begin,
                     /*overwrite=*/false);
      step_launched[ready] = 1;
      if (!step.sends.empty()) packing_step = ready;
    }
  }

  /*!
   * \brief Test each pending receive without blocking, which also lets MPI
   * progress the messages.
   */
  void TestReceives() {
#if MPI_PARALLEL_ENABLED
    for (std::size_t b = 0; b < recv_batches.size(); ++b) {
      if (recv_done[b]) continue;
      int flag;
      MPI_Test(&recv_reqs[b], &flag, MPI_STATUS_IGNORE);
      recv_done[b] = static_cast<char>(flag);
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

  // A slot of a message, on either end: the face of MeshBlock gid, sent to or
  // received from rank, on the face plane along the scan. Sorting by (rank,
  // plane, gid) groups the slots into batches identically on both ends
  struct PlanPort {
    int rank;
    std::int64_t plane;
    int gid;
    bool quarter;        // a finer sender's face, restricted to a quarter
    std::size_t offset;  // start in sendbuf or recvbuf, in Reals
    int batch;           // index in send_batches or recv_batches

    bool operator<(const PlanPort& o) const {
      return std::tie(rank, plane, gid) < std::tie(o.rank, o.plane, o.gid);
    }
    bool operator==(const PlanPort& o) const {
      return rank == o.rank && plane == o.plane && gid == o.gid;
    }
  };

  // One local MeshBlock's part of the plan. Its upstream sum is the sum of
  // its cone, in the local pass, and its anchors, once they are complete
  struct BlockPlan {
    std::vector<Neighbor> up, down;  // its upstream and downstream neighbours
    // The line totals of the run of local MeshBlocks at this MeshBlock's level
    // directly upstream of it, which map to it unchanged. A run of L
    // MeshBlocks so has L(L-1)/2 of these Sources, a cost that grows
    // quadratically with the length of the runs on a rank
    std::vector<Source> cone;
    // The upstream neighbours of the first MeshBlock of that run: remote ones
    // read from recvbuf, local ones (finer or coarser) as their outgoing
    // totals. Each maps to this MeshBlock by one neighbour map
    std::vector<Source> anchors;
    std::vector<int> anchor_recvs;  // the receive batches the anchors read
    std::vector<int> out_ports;     // its ports in PlanDraft::out_ports
    int step = -1;                  // the step that sums its anchors, if any
    bool critical = false;  // whether another step or rank reads its total
  };

  // The host state of one plan build, passed between the steps of
  // BuildExchangePlan
  struct PlanDraft {
    std::vector<int> order;  // local MeshBlocks by increasing upstream plane
    std::vector<PlanPort> in_ports, out_ports;
    std::vector<BlockPlan> blocks;
    std::vector<int> step_of_send;  // step packing each send batch, -1: local
  };

  /*!
   * \brief Build the exchange plan for the current mesh and copy it to the
   * device, without fencing. Called only by Driver, on the first run and after
   * the mesh changes.
   *
   * The plan is built on the host from each local MeshBlock's own neighbour
   * data (nghbr, mb_bcs, and lloc_eachmb for its neighbours), so building it
   * is independent of the size of the mesh and the number of ranks. Copying it
   * to the device is not: the whole allocated arrays are copied, and
   * MaxSources grows with the number of MeshBlocks along a line through the
   * mesh at max_level (up to nmb). Both sides of a message derive the same
   * layout, so no communication is needed.
   *
   * Each local MeshBlock B gets a cone and anchors (see BlockPlan). B's
   * outgoing total is complete once its anchors are summed. Other ranks need
   * the outgoing totals of MeshBlocks with a remote downstream neighbour
   * (out-ports, one per MeshBlock and rank), which a rank receives once per
   * distinct remote MeshBlock (in-ports).
   */
  void BuildExchangePlan() {
    PlanDraft draft;
    FindNeighbors(draft);
    LayOutPorts(draft);
    FindSources(draft);
    BuildSteps(draft);
    BuildItems(draft);

    // Asynchronously copy the plan arrays to the device. The host and device
    // arrays are allocated with the same sizes, so their extents match
    Kokkos::deep_copy(exec_space, items, h_items);
    Kokkos::deep_copy(exec_space, sources, h_sources);
    Kokkos::deep_copy(exec_space, ports, h_ports);

    // One MPI request per batch
#if MPI_PARALLEL_ENABLED
    send_reqs.assign(send_batches.size(), MPI_REQUEST_NULL);
    recv_reqs.assign(recv_batches.size(), MPI_REQUEST_NULL);
#endif  // MPI_PARALLEL_ENABLED

    // Update the mesh configuration that this plan was built for
    plan_mesh_seq = pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq();
  }

  /*!
   * \brief The global coordinate of a MeshBlock's upstream or downstream face
   * along the scan, in MeshBlocks of level max_level (the deepest level the
   * mesh may reach), negated for suffix scans so the totals always move to
   * larger planes. Both ends of a face agree on it.
   */
  std::int64_t Plane(const int gid, const bool downstream) const {
    const Mesh* pm = pmy_pack->pmesh;
    const LogicalLocation& loc = pm->lloc_eachmb[gid];
    const std::int64_t lx =
        SelectWithDirection<std::int64_t>(Dir, loc.lx1, loc.lx2, loc.lx3);
    const int shift = pm->max_level - loc.level;
    if (Kind == ScanKind::Prefix) return (lx + downstream) << shift;
    return -((lx + !downstream) << shift);
  }

  /*!
   * \brief Each local MeshBlock's upstream and downstream face neighbours, and
   * the order to visit the MeshBlocks in: by increasing plane of their
   * upstream face, so all of a MeshBlock's local upstream MeshBlocks come
   * before it.
   */
  void FindNeighbors(PlanDraft& draft) const {
    const int nmb_local = pmy_pack->nmb_thispack;
    const int first_gid = pmy_pack->gids;
    draft.blocks.resize(nmb_local);
    for (int m = 0; m < nmb_local; ++m) {
      draft.blocks[m].up = FaceNeighbors(m, /*upstream=*/true);
      draft.blocks[m].down = FaceNeighbors(m, /*upstream=*/false);
    }
    draft.order.resize(nmb_local);
    std::iota(draft.order.begin(), draft.order.end(), 0);
    std::stable_sort(
        draft.order.begin(), draft.order.end(), [&](const int a, const int b) {
          return Plane(first_gid + a, false) < Plane(first_gid + b, false);
        });
  }

  /*!
   * \brief The ports and their batches. In-ports: each distinct remote
   * upstream neighbour, a finer one sending a quarter face. Out-ports: each
   * local MeshBlock once per rank of its remote downstream neighbours.
   */
  void LayOutPorts(PlanDraft& draft) {
    const int nmb_local = pmy_pack->nmb_thispack;
    const int first_gid = pmy_pack->gids;
    const int my_rank = global_variable::my_rank;
    const LogicalLocation* lloc = pmy_pack->pmesh->lloc_eachmb;
    for (int m = 0; m < nmb_local; ++m) {
      const int gid = first_gid + m;
      for (const Neighbor& nb : draft.blocks[m].up) {
        if (nb.rank == my_rank) continue;
        draft.in_ports.push_back({nb.rank, Plane(gid, false), nb.gid,
                                  nb.level > lloc[gid].level, 0, 0});
      }
      for (const Neighbor& nb : draft.blocks[m].down) {
        if (nb.rank == my_rank) continue;
        draft.out_ports.push_back(
            {nb.rank, Plane(gid, true), gid, nb.level < lloc[gid].level, 0, 0});
      }
    }
    LayOutBatches(draft.in_ports, recv_batches);
    LayOutBatches(draft.out_ports, send_batches);
    for (std::size_t p = 0; p < draft.out_ports.size(); ++p) {
      draft.blocks[draft.out_ports[p].gid - first_gid].out_ports.push_back(
          static_cast<int>(p));
    }
  }

  /*!
   * \brief Sort and deduplicate the ports, so they are grouped into batches by
   * rank and plane in the same order on both ends, lay the batches out in
   * Reals and set each port's offset and batch. The tag is the plane's index
   * among this pair's planes.
   */
  void LayOutBatches(std::vector<PlanPort>& ports,
                     std::vector<Batch>& batches) const {
    std::sort(ports.begin(), ports.end());
    ports.erase(std::unique(ports.begin(), ports.end()), ports.end());
    batches.clear();
    std::size_t total = 0;
    for (std::size_t p = 0; p < ports.size(); ++p) {
      PlanPort& port = ports[p];
      const bool same_rank = p > 0 && port.rank == ports[p - 1].rank;
      if (!same_rank || port.plane != ports[p - 1].plane) {
        batches.push_back({port.rank, same_rank ? batches.back().tag + 1 : 0,
                           total, 0, std::vector<int>()});
      }
      const std::size_t size =
          std::size_t{1} * Num_Fields * (port.quarter ? h1 * h2 : n1 * n2);
      port.offset = total;
      port.batch = static_cast<int>(batches.size()) - 1;
      batches.back().count += size;
      batches.back().gids.push_back(port.gid);
      total += size;
    }
  }

  /*!
   * \brief Each MeshBlock's cone and anchors. A MeshBlock whose only upstream
   * neighbour is a local MeshBlock at its level extends that MeshBlock's run:
   * it takes its cone, plus that MeshBlock's own line totals, and its anchors,
   * which map to it the same way.
   */
  void FindSources(PlanDraft& draft) const {
    const int first_gid = pmy_pack->gids;
    const int my_rank = global_variable::my_rank;
    const LogicalLocation* lloc = pmy_pack->pmesh->lloc_eachmb;

    // The in-port of each remote gid, found by binary search
    std::vector<std::pair<int, int>> in_port_of;
    for (std::size_t p = 0; p < draft.in_ports.size(); ++p) {
      in_port_of.emplace_back(draft.in_ports[p].gid, static_cast<int>(p));
    }
    std::sort(in_port_of.begin(), in_port_of.end());
    auto InPort = [&](const int gid) -> const PlanPort& {
      const int p = std::lower_bound(in_port_of.begin(), in_port_of.end(),
                                     std::make_pair(gid, -1))
                        ->second;
      return draft.in_ports[p];
    };

    for (const int m : draft.order) {
      BlockPlan& block = draft.blocks[m];
      const LogicalLocation& loc = lloc[first_gid + m];
      const std::vector<Neighbor>& up = block.up;
      if (up.size() == 1 && up[0].rank == my_rank && up[0].level == loc.level) {
        const int u = up[0].gid - first_gid;
        const BlockPlan& prev = draft.blocks[u];
        Source src;
        src.m = u;
        block.cone = prev.cone;
        block.cone.push_back(src);
        block.anchors = prev.anchors;
        block.anchor_recvs = prev.anchor_recvs;
        continue;
      }
      for (const Neighbor& nb : up) {
        Source src;
        src.dlevel = nb.level - loc.level;
        if (src.dlevel > 0) {
          // A finer neighbour covers the quadrant given by its sub-slot
          src.off1 = (nb.sub % 2) * h1;
          src.off2 = (nb.sub / 2) * h2;
        } else if (src.dlevel < 0) {
          // This MeshBlock covers the quadrant of the coarser one given by its
          // own position, 0 along an unused direction
          src.off1 =
              (SelectWithDirection(t1_dir, loc.lx1, loc.lx2, loc.lx3) & 1) * h1;
          src.off2 =
              (SelectWithDirection(t2_dir, loc.lx1, loc.lx2, loc.lx3) & 1) * h2;
        }
        if (nb.rank == my_rank) {
          src.m = nb.gid - first_gid;
          src.out = true;
        } else {
          const PlanPort& port = InPort(nb.gid);
          src.offset = port.offset;
          block.anchor_recvs.push_back(port.batch);
          // A finer remote neighbour sends its outgoing total already
          // restricted, a quarter face at this level
          src.quarter = port.quarter;
          if (src.quarter) src.dlevel = 0;
        }
        block.anchors.push_back(src);
      }
    }
  }

  /*!
   * \brief Group the MeshBlocks with anchors into steps by the receives and
   * steps they wait for, which come from MeshBlocks visited earlier, and give
   * each send batch to the launch that packs it: the local pass if none of its
   * MeshBlocks has anchors, else their one step or a step after all of theirs.
   * A MeshBlock is critical if another step or rank reads its outgoing total;
   * the rest are summed in one launch after every step.
   */
  void BuildSteps(PlanDraft& draft) {
    const int first_gid = pmy_pack->gids;
    for (BlockPlan& block : draft.blocks) {
      if (!block.out_ports.empty()) block.critical = true;
      for (const Source& src : block.anchors) {
        if (src.out) draft.blocks[src.m].critical = true;
      }
    }

    steps.clear();
    std::map<std::pair<std::vector<int>, std::vector<int>>, int> step_of_key;
    auto FindStep = [&](std::vector<int> recvs, std::vector<int> after) {
      for (std::vector<int>* v : {&recvs, &after}) {
        std::sort(v->begin(), v->end());
        v->erase(std::unique(v->begin(), v->end()), v->end());
      }
      auto key = std::make_pair(recvs, after);
      auto it = step_of_key.find(key);
      if (it != step_of_key.end()) return it->second;
      steps.push_back({recvs, after, 0, 0, std::vector<int>()});
      return step_of_key[key] = static_cast<int>(steps.size()) - 1;
    };
    for (const int m : draft.order) {
      BlockPlan& block = draft.blocks[m];
      if (block.anchors.empty()) continue;
      std::vector<int> after;
      for (const Source& src : block.anchors) {
        if (src.out && draft.blocks[src.m].step >= 0) {
          after.push_back(draft.blocks[src.m].step);
        }
      }
      block.step = FindStep(block.anchor_recvs, after);
    }

    first_sends.clear();
    draft.step_of_send.assign(send_batches.size(), -1);
    for (std::size_t b = 0; b < send_batches.size(); ++b) {
      std::vector<int> after;
      for (const int gid : send_batches[b].gids) {
        const int step = draft.blocks[gid - first_gid].step;
        if (step >= 0) after.push_back(step);
      }
      std::sort(after.begin(), after.end());
      after.erase(std::unique(after.begin(), after.end()), after.end());
      if (after.empty()) {
        first_sends.push_back(static_cast<int>(b));
        continue;
      }
      const int step = (after.size() == 1) ? after[0] : FindStep({}, after);
      draft.step_of_send[b] = step;
      steps[step].sends.push_back(static_cast<int>(b));
    }
  }

  /*!
   * \brief The items, Sources and Ports of every launch, in launch order: the
   * local pass (the MeshBlocks packed by first_sends first), each step's
   * MeshBlocks, then the deferred ones; copied to the host plan arrays.
   */
  void BuildItems(PlanDraft& draft) {
    const int nmb_local = pmy_pack->nmb_thispack;
    const int first_gid = pmy_pack->gids;
    std::vector<Item> item_list;
    std::vector<Source> source_list;
    std::vector<Port> port_list;
    // Adds an item for MeshBlock m with the given Sources and its ports in the
    // send batches packed by step (-1 for the local pass)
    auto AddItem = [&](const int m, const std::vector<Source>& srcs,
                       const int step) {
      Item item = {m, static_cast<int>(source_list.size()), 0,
                   static_cast<int>(port_list.size()), 0};
      source_list.insert(source_list.end(), srcs.begin(), srcs.end());
      for (const int p : draft.blocks[m].out_ports) {
        const PlanPort& port = draft.out_ports[p];
        if (draft.step_of_send[port.batch] == step) {
          port_list.push_back({port.offset, port.quarter});
        }
      }
      item.src_end = static_cast<int>(source_list.size());
      item.port_end = static_cast<int>(port_list.size());
      item_list.push_back(item);
    };

    // The local pass, MeshBlocks packed by it first
    std::vector<char> packs_first(nmb_local, 0);
    for (const int b : first_sends) {
      for (const int gid : send_batches[b].gids) {
        packs_first[gid - first_gid] = 1;
      }
    }
    for (int m = 0; m < nmb_local; ++m) {
      if (packs_first[m]) AddItem(m, draft.blocks[m].cone, -1);
    }
    n_first_items = static_cast<int>(item_list.size());
    for (int m = 0; m < nmb_local; ++m) {
      if (!packs_first[m]) AddItem(m, draft.blocks[m].cone, -1);
    }
    n_local_items = static_cast<int>(item_list.size());

    // Each step's MeshBlocks that something reads, and the others its sends
    // pack, which are already complete
    std::vector<std::vector<int>> step_blocks(steps.size());
    for (const int m : draft.order) {
      const int step = draft.blocks[m].step;
      if (step >= 0) step_blocks[step].push_back(m);
    }
    std::vector<int> item_step(nmb_local, -1);  // the step m has an item in
    for (int s = 0; s < static_cast<int>(steps.size()); ++s) {
      steps[s].item_begin = static_cast<int>(item_list.size());
      for (const int m : step_blocks[s]) {
        if (!draft.blocks[m].critical) continue;
        AddItem(m, draft.blocks[m].anchors, s);
        item_step[m] = s;
      }
      for (const int b : steps[s].sends) {
        for (const int gid : send_batches[b].gids) {
          const int m = gid - first_gid;
          if (item_step[m] == s) continue;
          AddItem(m, {}, s);
          item_step[m] = s;
        }
      }
      steps[s].item_end = static_cast<int>(item_list.size());
    }

    // Deferred MeshBlocks aren't critical, so they have no out-ports
    deferred_begin = static_cast<int>(item_list.size());
    for (const int m : draft.order) {
      const BlockPlan& block = draft.blocks[m];
      if (!block.anchors.empty() && !block.critical) {
        AddItem(m, block.anchors, -1);
      }
    }
    deferred_end = static_cast<int>(item_list.size());

    CheckPlanFits(item_list.size(), source_list.size(), port_list.size());
    std::copy(item_list.begin(), item_list.end(), h_items.data());
    std::copy(source_list.begin(), source_list.end(), h_sources.data());
    std::copy(port_list.begin(), port_list.end(), h_ports.data());
  }

  /*!
   * \brief Local MeshBlock m's upstream or downstream face neighbours. Only
   * a face with a real neighbour, BoundaryFlag::block in mb_bcs, has any:
   * nghbr also holds neighbours across periodic boundaries, which must not
   * be followed, and mb_bcs holds the mesh's own boundary flag on every
   * domain edge, including both faces along an unused direction, where nghbr
   * has no entries.
   */
  std::vector<Neighbor> FaceNeighbors(const int m, const bool upstream) const {
    std::vector<Neighbor> list;
    // -1 for the face at the lower edge along the scan direction, 1 for upper
    const int side = (upstream == (Kind == ScanKind::Prefix)) ? -1 : 1;
    if (pmy_pack->pmb->mb_bcs.h_view(m, 2 * s_axis + (side > 0)) !=
        BoundaryFlag::block) {
      return list;
    }
    // The face's 4 sub-slots in nghbr, by quadrant (t1) + 2 (t2) for finer
    // neighbours, or this MeshBlock's own quadrant for a coarser one
    const int slot =
        NeighborIndex((s_axis == 0) ? side : 0, (s_axis == 1) ? side : 0,
                      (s_axis == 2) ? side : 0, 0, 0);
    for (int sub = 0; sub < 4; ++sub) {
      const NeighborBlock& nb = pmy_pack->pmb->nghbr.h_view(m, slot + sub);
      if (nb.gid >= 0) list.push_back({nb.gid, nb.rank, nb.lev, sub});
    }
    return list;
  }

  //--------------------------------------------------------------------------------------
  // Plan bounds. The plan and message arrays are allocated once, at these
  // bounds on the plan of nmb MeshBlocks, so a plan rebuild never allocates

  /*!
   * \brief The most face neighbours a MeshBlock has on one side: r1 x r2
   * finer ones on a multilevel mesh, else one.
   */
  int MaxFaceNeighbors() const {
    return pmy_pack->pmesh->multilevel ? r1 * r2 : 1;
  }

  /*!
   * \brief The Reals of a full face of every field.
   */
  std::size_t FaceReals() const {
    return std::size_t{1} * Num_Fields * n1 * n2;
  }

  /*!
   * \brief Ports: one per MeshBlock and rank of its remote downstream
   * neighbours, so at most one per neighbour. Each sends at most a full face,
   * so sendbuf holds at most MaxPorts() faces. recvbuf holds at most one face
   * per MeshBlock: a neighbour at its level or coarser sends a full face, and
   * its r1 x r2 finer neighbours each send a quarter face.
   */
  std::size_t MaxPorts() const {
    return std::size_t{1} * nmb * MaxFaceNeighbors();
  }

  /*!
   * \brief Items: every MeshBlock in the local pass, at most one more per
   * MeshBlock that sums its anchors, and at most one pack-only item per Port.
   */
  std::size_t MaxItems() const { return std::size_t{2} * nmb + MaxPorts(); }

  /*!
   * \brief Sources: a run of L MeshBlocks has L(L-1)/2 cone Sources (see
   * BlockPlan), at most (L_max - 1) / 2 per MeshBlock, where a run is no
   * longer than nmb or a line of MeshBlocks through the mesh at max_level;
   * and each MeshBlock sums at most one anchor per upstream neighbour.
   */
  std::size_t MaxSources() const {
    const Mesh* pm = pmy_pack->pmesh;
    const std::size_t line =
        static_cast<std::size_t>(SelectWithDirection(
            Dir, pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3))
        << (pm->max_level - pm->root_level);
    const std::size_t max_run = std::min(static_cast<std::size_t>(nmb), line);
    return nmb * (max_run - 1) / 2 + std::size_t{1} * nmb * MaxFaceNeighbors();
  }

  /*!
   * \brief Abort if the plan doesn't fit in the arrays allocated for it,
   * which would mean a bound above is wrong.
   */
  void CheckPlanFits(const std::size_t n_items, const std::size_t n_sources,
                     const std::size_t n_ports) const {
    const auto Reals = [](const std::vector<Batch>& batches) {
      return batches.empty() ? std::size_t{0}
                             : batches.back().offset + batches.back().count;
    };
    const auto Check = [](const char* name, const std::size_t size,
                          const std::size_t extent) {
      if (size > extent) {
        FatalError(std::string("LineScan exchange plan needs ") +
                   std::to_string(size) + " entries of " + name +
                   ", more than the " + std::to_string(extent) + " allocated");
      }
    };
    Check("items", n_items, items.extent(0));
    Check("sources", n_sources, sources.extent(0));
    Check("ports", n_ports, ports.extent(0));
    Check("sendbuf", Reals(send_batches), sendbuf.extent(0));
    Check("recvbuf", Reals(recv_batches), recvbuf.extent(0));
  }

  //--------------------------------------------------------------------------------------
  // Helpers

  /*!
   * \brief The factor by which a face is restricted along a transverse
   * direction of n cells: 2 where it can map to a neighbouring level and 1
   * along an unused direction. Mesh refinement needs even MeshBlocks, so an
   * odd n > 1 only occurs on uniform meshes, where nothing is restricted but
   * ImplAccumulate must still visit every face cell; hence the test is for
   * evenness rather than n > 1.
   */
  static int RestrictFactor(const int n) { return (n % 2 == 0) ? 2 : 1; }

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
#endif  // UTILS_LINE_SCAN_HPP_
